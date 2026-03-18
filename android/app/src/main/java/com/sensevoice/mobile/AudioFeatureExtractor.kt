package com.sensevoice.mobile

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin

/**
 * Extracts FBank (Mel filterbank) features from raw audio and applies
 * Low Frame Rate (LFR) merging and CMVN normalization for SenseVoice model input.
 *
 * Audio preprocessing pipeline:
 * 1. Preemphasis (coefficient 0.97)
 * 2. Framing (25ms window, 10ms shift)
 * 3. Hann windowing
 * 4. FFT (power spectrum)
 * 5. Mel filterbank (80 bins)
 * 6. Log energy
 * 7. LFR (merge 7 frames, skip every 6) -> 560-dim features
 * 8. CMVN normalization (subtract mean, scale by variance)
 */
class AudioFeatureExtractor(
    private val sampleRate: Int = SAMPLE_RATE,
    private val numMelBins: Int = NUM_MEL_BINS,
    private val frameLengthMs: Int = FRAME_LENGTH_MS,
    private val frameShiftMs: Int = FRAME_SHIFT_MS,
    private val preemphCoeff: Float = PREEMPH_COEFF,
    private val lfrM: Int = LFR_M,
    private val lfrN: Int = LFR_N
) {
    private val frameLength = sampleRate * frameLengthMs / 1000
    private val frameShift = sampleRate * frameShiftMs / 1000
    private val fftSize = nextPowerOf2(frameLength)
    private val hannWindow = createHannWindow(frameLength)
    private val melFilterbank = createMelFilterbank(numMelBins, fftSize, sampleRate)

    /**
     * Extract features from raw 16kHz PCM audio samples.
     * Returns a 2D array of shape [numFrames, featureDim] where featureDim = numMelBins * lfrM.
     */
    fun extract(samples: FloatArray): Array<FloatArray> {
        val preemphasized = applyPreemphasis(samples)
        val fbank = computeFbank(preemphasized)
        val lfrFeatures = applyLFR(fbank)
        return lfrFeatures
    }

    /**
     * Extract features and apply CMVN normalization.
     * @param cmvnMean Mean values for normalization (dim = numMelBins * lfrM)
     * @param cmvnIstd Inverse standard deviation for normalization
     */
    fun extractWithCMVN(
        samples: FloatArray,
        cmvnMean: FloatArray,
        cmvnIstd: FloatArray
    ): Array<FloatArray> {
        val features = extract(samples)
        applyCMVN(features, cmvnMean, cmvnIstd)
        return features
    }

    private fun applyPreemphasis(samples: FloatArray): FloatArray {
        if (samples.isEmpty()) return samples
        val result = FloatArray(samples.size)
        result[0] = samples[0]
        for (i in 1 until samples.size) {
            result[i] = samples[i] - preemphCoeff * samples[i - 1]
        }
        return result
    }

    private fun computeFbank(samples: FloatArray): Array<FloatArray> {
        val numFrames = if (samples.size < frameLength) 0
        else 1 + (samples.size - frameLength) / frameShift

        if (numFrames == 0) return emptyArray()

        val fbank = Array(numFrames) { FloatArray(numMelBins) }

        for (frame in 0 until numFrames) {
            val start = frame * frameShift
            val windowed = FloatArray(fftSize)
            for (i in 0 until frameLength) {
                windowed[i] = if (start + i < samples.size) {
                    samples[start + i] * hannWindow[i]
                } else {
                    0f
                }
            }

            val (real, imag) = fft(windowed)

            val powerSpec = FloatArray(fftSize / 2 + 1)
            for (i in powerSpec.indices) {
                powerSpec[i] = real[i] * real[i] + imag[i] * imag[i]
            }

            for (mel in 0 until numMelBins) {
                var sum = 0f
                for (k in powerSpec.indices) {
                    sum += melFilterbank[mel][k] * powerSpec[k]
                }
                fbank[frame][mel] = ln(max(sum, 1e-10f))
            }
        }

        return fbank
    }

    private fun applyLFR(fbank: Array<FloatArray>): Array<FloatArray> {
        if (fbank.isEmpty()) return emptyArray()

        val numFrames = fbank.size
        val featureDim = fbank[0].size
        val lfrFrames = (numFrames + lfrN - 1) / lfrN
        val lfrDim = featureDim * lfrM

        val result = Array(lfrFrames) { FloatArray(lfrDim) }

        for (i in 0 until lfrFrames) {
            val centerFrame = i * lfrN
            for (j in 0 until lfrM) {
                val srcFrame = (centerFrame + j).coerceIn(0, numFrames - 1)
                System.arraycopy(fbank[srcFrame], 0, result[i], j * featureDim, featureDim)
            }
        }

        return result
    }

    private fun applyCMVN(
        features: Array<FloatArray>,
        mean: FloatArray,
        istd: FloatArray
    ) {
        for (frame in features) {
            for (i in frame.indices) {
                frame[i] = (frame[i] + mean[i]) * istd[i]
            }
        }
    }

    companion object {
        const val SAMPLE_RATE = 16000
        const val NUM_MEL_BINS = 80
        const val FRAME_LENGTH_MS = 25
        const val FRAME_SHIFT_MS = 10
        const val PREEMPH_COEFF = 0.97f
        const val LFR_M = 7
        const val LFR_N = 6
        const val FEATURE_DIM = NUM_MEL_BINS * LFR_M // 560

        private fun nextPowerOf2(n: Int): Int {
            var v = n - 1
            v = v or (v shr 1)
            v = v or (v shr 2)
            v = v or (v shr 4)
            v = v or (v shr 8)
            v = v or (v shr 16)
            return v + 1
        }

        private fun createHannWindow(size: Int): FloatArray {
            return FloatArray(size) { i ->
                (0.5 * (1.0 - cos(2.0 * PI * i / (size - 1)))).toFloat()
            }
        }

        private fun hzToMel(hz: Float): Float {
            return 2595.0f * log10(1.0f + hz / 700.0f)
        }

        private fun melToHz(mel: Float): Float {
            return 700.0f * (10.0f.pow(mel / 2595.0f) - 1.0f)
        }

        private fun createMelFilterbank(
            numMelBins: Int,
            fftSize: Int,
            sampleRate: Int
        ): Array<FloatArray> {
            val numBins = fftSize / 2 + 1
            val lowFreq = 0f
            val highFreq = sampleRate / 2f

            val lowMel = hzToMel(lowFreq)
            val highMel = hzToMel(highFreq)

            val melPoints = FloatArray(numMelBins + 2) { i ->
                melToHz(lowMel + i * (highMel - lowMel) / (numMelBins + 1))
            }

            val binPoints = IntArray(numMelBins + 2) { i ->
                floor(melPoints[i] * fftSize / sampleRate).toInt().coerceIn(0, numBins - 1)
            }

            val filterbank = Array(numMelBins) { FloatArray(numBins) }
            for (m in 0 until numMelBins) {
                for (k in binPoints[m]..binPoints[m + 1]) {
                    if (binPoints[m + 1] != binPoints[m]) {
                        filterbank[m][k] =
                            (k - binPoints[m]).toFloat() / (binPoints[m + 1] - binPoints[m])
                    }
                }
                for (k in binPoints[m + 1]..binPoints[m + 2]) {
                    if (binPoints[m + 2] != binPoints[m + 1]) {
                        filterbank[m][k] =
                            (binPoints[m + 2] - k).toFloat() / (binPoints[m + 2] - binPoints[m + 1])
                    }
                }
            }
            return filterbank
        }

        /**
         * Simple in-place radix-2 FFT. Input length must be a power of 2.
         * Returns (realPart, imagPart).
         */
        private fun fft(input: FloatArray): Pair<FloatArray, FloatArray> {
            val n = input.size
            val real = input.copyOf()
            val imag = FloatArray(n)

            // Bit-reversal permutation
            var j = 0
            for (i in 0 until n) {
                if (i < j) {
                    val tmpR = real[i]; real[i] = real[j]; real[j] = tmpR
                }
                var m = n / 2
                while (m >= 1 && j >= m) {
                    j -= m
                    m /= 2
                }
                j += m
            }

            // Cooley-Tukey butterfly
            var step = 2
            while (step <= n) {
                val halfStep = step / 2
                val angleStep = -2.0 * PI / step
                for (i in 0 until n step step) {
                    for (k in 0 until halfStep) {
                        val angle = angleStep * k
                        val wr = cos(angle).toFloat()
                        val wi = sin(angle).toFloat()
                        val idx1 = i + k
                        val idx2 = i + k + halfStep
                        val tReal = wr * real[idx2] - wi * imag[idx2]
                        val tImag = wr * imag[idx2] + wi * real[idx2]
                        real[idx2] = real[idx1] - tReal
                        imag[idx2] = imag[idx1] - tImag
                        real[idx1] = real[idx1] + tReal
                        imag[idx1] = imag[idx1] + tImag
                    }
                }
                step *= 2
            }

            return Pair(real, imag)
        }
    }
}
