package com.sensevoice.mobile

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.core.app.ActivityCompat
import android.content.Context
import java.io.ByteArrayOutputStream

/**
 * Records audio from the device microphone as 16kHz 16-bit mono PCM.
 *
 * Usage:
 * ```
 * val recorder = AudioRecorder(context)
 * recorder.startRecording()
 * // ... wait ...
 * val samples = recorder.stopRecording()
 * ```
 */
class AudioRecorder(private val context: Context) {
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingThread: Thread? = null
    private val audioData = ByteArrayOutputStream()

    /**
     * Starts recording audio from the microphone.
     * Requires RECORD_AUDIO permission.
     * @return true if recording started successfully
     */
    fun startRecording(): Boolean {
        if (isRecording) return false

        if (ActivityCompat.checkSelfPermission(
                context, Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            return false
        }

        val bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            return false
        }

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize * 2
        )

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            audioRecord?.release()
            audioRecord = null
            return false
        }

        audioData.reset()
        isRecording = true
        audioRecord?.startRecording()

        recordingThread = Thread {
            val buffer = ByteArray(bufferSize)
            while (isRecording) {
                val read = audioRecord?.read(buffer, 0, buffer.size) ?: -1
                if (read > 0) {
                    synchronized(audioData) {
                        audioData.write(buffer, 0, read)
                    }
                }
            }
        }.apply {
            name = "AudioRecorder"
            start()
        }

        return true
    }

    /**
     * Stops recording and returns the recorded audio as float samples
     * normalized to [-1.0, 1.0].
     */
    fun stopRecording(): FloatArray {
        isRecording = false
        recordingThread?.join(2000)
        recordingThread = null

        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null

        val bytes: ByteArray
        synchronized(audioData) {
            bytes = audioData.toByteArray()
        }

        return pcmBytesToFloat(bytes)
    }

    /** Returns true if currently recording. */
    fun getRecordingState(): Boolean = isRecording

    companion object {
        const val SAMPLE_RATE = 16000

        /**
         * Converts 16-bit PCM byte data to float samples in [-1.0, 1.0] range.
         */
        fun pcmBytesToFloat(bytes: ByteArray): FloatArray {
            val numSamples = bytes.size / 2
            val samples = FloatArray(numSamples)
            for (i in 0 until numSamples) {
                val low = bytes[i * 2].toInt() and 0xFF
                val high = bytes[i * 2 + 1].toInt()
                val sample = (high shl 8) or low
                samples[i] = sample / 32768.0f
            }
            return samples
        }
    }
}
