package com.sensevoice.mobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Wraps the SenseVoice ONNX model for inference on Android.
 *
 * Required assets in the app's assets directory:
 * - model.onnx: The exported SenseVoice ONNX model
 * - tokens.txt: Token vocabulary (format: "token id" per line)
 * - am.mvn: CMVN statistics (mean and inverse std deviation)
 *
 * The model accepts:
 * - speech_feats: [1, num_frames, 560] float32 (LFR+CMVN features)
 * - speech_lengths: [1] int64 (number of frames)
 *
 * And produces:
 * - logits: [1, num_frames, vocab_size] float32 (CTC logits)
 */
class SenseVoiceModel(private val context: Context) {
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var tokenDecoder: TokenDecoder? = null
    private var featureExtractor: AudioFeatureExtractor? = null
    private var cmvnMean: FloatArray? = null
    private var cmvnIstd: FloatArray? = null
    private var vocabSize: Int = 0
    private var isLoaded = false

    /**
     * Loads the model, vocabulary, and CMVN statistics from assets.
     * @throws RuntimeException if any required file is missing
     */
    fun load() {
        ortEnvironment = OrtEnvironment.getEnvironment()

        val sessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(4)
        }

        val modelBytes = context.assets.open(MODEL_FILE).readBytes()
        ortSession = ortEnvironment?.createSession(modelBytes, sessionOptions)

        val vocabContent = context.assets.open(TOKENS_FILE).bufferedReader().readText()
        val vocabulary = TokenDecoder.loadVocabulary(vocabContent)
        tokenDecoder = TokenDecoder(vocabulary)
        vocabSize = vocabulary.size

        loadCMVN()

        featureExtractor = AudioFeatureExtractor()
        isLoaded = true
    }

    /**
     * Runs inference on raw audio samples (16kHz, mono, float [-1, 1]).
     * @return DecodingResult with language, emotion, event, and text
     */
    fun inference(audioSamples: FloatArray): TokenDecoder.DecodingResult {
        check(isLoaded) { "Model not loaded. Call load() first." }

        val features = if (cmvnMean != null && cmvnIstd != null) {
            featureExtractor!!.extractWithCMVN(audioSamples, cmvnMean!!, cmvnIstd!!)
        } else {
            featureExtractor!!.extract(audioSamples)
        }

        if (features.isEmpty()) {
            return TokenDecoder.DecodingResult(text = "")
        }

        val numFrames = features.size
        val featureDim = features[0].size

        val env = ortEnvironment!!
        val session = ortSession!!

        val featsFlat = FloatArray(numFrames * featureDim)
        for (i in features.indices) {
            System.arraycopy(features[i], 0, featsFlat, i * featureDim, featureDim)
        }

        val featsShape = longArrayOf(1, numFrames.toLong(), featureDim.toLong())
        val featsTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(featsFlat),
            featsShape
        )

        val lengthsShape = longArrayOf(1)
        val lengthsTensor = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(longArrayOf(numFrames.toLong())),
            lengthsShape
        )

        val inputs = mutableMapOf<String, OnnxTensor>(
            "speech" to featsTensor,
            "speech_lengths" to lengthsTensor
        )

        val inputNames = session.inputNames
        if ("language" in inputNames) {
            val langTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(longArrayOf(0)),
                longArrayOf(1)
            )
            inputs["language"] = langTensor
        }
        if ("text_norm" in inputNames) {
            val itnTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(longArrayOf(15)),
                longArrayOf(1)
            )
            inputs["text_norm"] = itnTensor
        }

        val results = session.run(inputs)

        val outputTensor = results[0] as OnnxTensor
        val outputShape = outputTensor.info.shape
        val outFrames = outputShape[1].toInt()
        val outVocabSize = outputShape[2].toInt()
        val logits = outputTensor.floatBuffer
        val logitsArray = FloatArray(outFrames * outVocabSize)
        logits.get(logitsArray)

        val result = tokenDecoder!!.ctcGreedyDecode(logitsArray, outFrames, outVocabSize)

        results.close()
        for (tensor in inputs.values) {
            tensor.close()
        }

        return result
    }

    /** Releases all resources. */
    fun release() {
        ortSession?.close()
        ortEnvironment?.close()
        ortSession = null
        ortEnvironment = null
        tokenDecoder = null
        featureExtractor = null
        isLoaded = false
    }

    /** Returns true if the model is loaded and ready. */
    fun isReady(): Boolean = isLoaded

    private fun loadCMVN() {
        try {
            val reader = BufferedReader(InputStreamReader(context.assets.open(CMVN_FILE)))
            val lines = reader.readLines()
            reader.close()

            if (lines.size >= 2) {
                cmvnMean = parseFloatLine(lines[0])
                cmvnIstd = parseFloatLine(lines[1])
            }
        } catch (_: Exception) {
            // CMVN file is optional; skip normalization if not present
            cmvnMean = null
            cmvnIstd = null
        }
    }

    private fun parseFloatLine(line: String): FloatArray {
        return line.trim()
            .removePrefix("<AddShift>").removePrefix("<Rescale>")
            .removePrefix("<LearnRateFixed>")
            .trim()
            .split("\\s+".toRegex())
            .filter { it.isNotEmpty() }
            .mapNotNull { it.toFloatOrNull() }
            .toFloatArray()
    }

    companion object {
        private const val MODEL_FILE = "model.onnx"
        private const val TOKENS_FILE = "tokens.txt"
        private const val CMVN_FILE = "am.mvn"

        /**
         * Checks whether all required model files exist in assets.
         */
        fun hasModelFiles(context: Context): Boolean {
            return try {
                val assets = context.assets.list("") ?: return false
                MODEL_FILE in assets && TOKENS_FILE in assets
            } catch (_: Exception) {
                false
            }
        }
    }
}
