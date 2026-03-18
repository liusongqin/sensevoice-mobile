package com.sensevoice.mobile

import android.content.Context
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Utility class for loading and decoding WAV audio files into float samples.
 * Supports standard PCM WAV format (16-bit, mono/stereo) and resamples to 16kHz.
 */
object AudioFileReader {

    /**
     * Reads a WAV file from an InputStream and returns float samples at 16kHz mono.
     * Supports 16-bit PCM WAV files.
     */
    fun readWav(inputStream: InputStream): FloatArray {
        val bytes = inputStream.readBytes()
        inputStream.close()

        if (bytes.size < 44) {
            throw IllegalArgumentException("Invalid WAV file: too small")
        }

        val header = String(bytes, 0, 4)
        if (header != "RIFF") {
            throw IllegalArgumentException("Invalid WAV file: missing RIFF header")
        }

        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // Parse WAV header
        buffer.position(20)
        val audioFormat = buffer.short.toInt()
        val numChannels = buffer.short.toInt()
        val sampleRate = buffer.int
        buffer.int // byteRate
        buffer.short // blockAlign
        val bitsPerSample = buffer.short.toInt()

        if (audioFormat != 1) {
            throw IllegalArgumentException("Unsupported WAV format: only PCM is supported")
        }

        // Find data chunk
        var dataOffset = 36
        while (dataOffset + 8 < bytes.size) {
            val chunkId = String(bytes, dataOffset, 4)
            val chunkSize = ByteBuffer.wrap(bytes, dataOffset + 4, 4)
                .order(ByteOrder.LITTLE_ENDIAN).int
            if (chunkId == "data") {
                dataOffset += 8
                break
            }
            dataOffset += 8 + chunkSize
        }

        val dataSize = bytes.size - dataOffset
        val bytesPerSample = bitsPerSample / 8
        val numSamples = dataSize / (bytesPerSample * numChannels)

        // Read samples
        val samples = FloatArray(numSamples)
        val dataBuffer = ByteBuffer.wrap(bytes, dataOffset, dataSize)
            .order(ByteOrder.LITTLE_ENDIAN)

        for (i in 0 until numSamples) {
            var sampleValue = 0f
            for (ch in 0 until numChannels) {
                val value = when (bitsPerSample) {
                    16 -> dataBuffer.short.toFloat() / 32768f
                    8 -> (dataBuffer.get().toInt() and 0xFF - 128).toFloat() / 128f
                    else -> {
                        dataBuffer.position(dataBuffer.position() + bytesPerSample)
                        0f
                    }
                }
                sampleValue += value
            }
            samples[i] = sampleValue / numChannels
        }

        // Resample to 16kHz if needed
        return if (sampleRate != AudioFeatureExtractor.SAMPLE_RATE) {
            resample(samples, sampleRate, AudioFeatureExtractor.SAMPLE_RATE)
        } else {
            samples
        }
    }

    /**
     * Reads raw PCM audio from an InputStream.
     * Assumes 16-bit, mono, 16kHz.
     */
    fun readRawPcm(inputStream: InputStream): FloatArray {
        val bytes = inputStream.readBytes()
        inputStream.close()
        return AudioRecorder.pcmBytesToFloat(bytes)
    }

    /**
     * Simple linear interpolation resampling.
     */
    private fun resample(samples: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        if (fromRate == toRate) return samples
        val ratio = fromRate.toDouble() / toRate
        val outputLength = (samples.size / ratio).toInt()
        val output = FloatArray(outputLength)

        for (i in output.indices) {
            val srcPos = i * ratio
            val srcIdx = srcPos.toInt()
            val frac = (srcPos - srcIdx).toFloat()
            output[i] = if (srcIdx + 1 < samples.size) {
                samples[srcIdx] * (1 - frac) + samples[srcIdx + 1] * frac
            } else {
                samples[srcIdx.coerceIn(0, samples.size - 1)]
            }
        }
        return output
    }
}
