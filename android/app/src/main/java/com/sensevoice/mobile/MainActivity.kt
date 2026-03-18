package com.sensevoice.mobile

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Main activity for the SenseVoice speech recognition app.
 *
 * Features:
 * - Record audio from microphone and transcribe
 * - Select audio files (WAV) for transcription
 * - Display recognized language, emotion, event, and text
 */
class MainActivity : AppCompatActivity() {
    private lateinit var tvStatus: TextView
    private lateinit var tvResult: TextView
    private lateinit var tvLanguage: TextView
    private lateinit var tvEmotion: TextView
    private lateinit var tvEvent: TextView
    private lateinit var tvLabelLang: TextView
    private lateinit var tvLabelEmotion: TextView
    private lateinit var tvLabelEvent: TextView
    private lateinit var tvLabelText: TextView
    private lateinit var btnRecord: MaterialButton
    private lateinit var btnSelectFile: MaterialButton
    private lateinit var progressBar: ProgressBar

    private var model: SenseVoiceModel? = null
    private var recorder: AudioRecorder? = null
    private var isRecording = false
    private val scope = CoroutineScope(Dispatchers.Main + Job())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        requestPermissions()
        loadModel()
    }

    private fun initViews() {
        tvStatus = findViewById(R.id.tvStatus)
        tvResult = findViewById(R.id.tvResult)
        tvLanguage = findViewById(R.id.tvLanguage)
        tvEmotion = findViewById(R.id.tvEmotion)
        tvEvent = findViewById(R.id.tvEvent)
        tvLabelLang = findViewById(R.id.tvLabelLang)
        tvLabelEmotion = findViewById(R.id.tvLabelEmotion)
        tvLabelEvent = findViewById(R.id.tvLabelEvent)
        tvLabelText = findViewById(R.id.tvLabelText)
        btnRecord = findViewById(R.id.btnRecord)
        btnSelectFile = findViewById(R.id.btnSelectFile)
        progressBar = findViewById(R.id.progressBar)

        btnRecord.setOnClickListener { toggleRecording() }
        btnSelectFile.setOnClickListener { selectAudioFile() }
    }

    private fun requestPermissions() {
        val permissions = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            permissions.add(Manifest.permission.RECORD_AUDIO)
        }
        if (permissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this, permissions.toTypedArray(), REQUEST_PERMISSIONS
            )
        }
    }

    private fun loadModel() {
        tvStatus.text = getString(R.string.status_loading_model)
        progressBar.visibility = View.VISIBLE
        btnRecord.isEnabled = false
        btnSelectFile.isEnabled = false

        scope.launch {
            try {
                val senseVoiceModel = SenseVoiceModel(this@MainActivity)
                withContext(Dispatchers.IO) {
                    senseVoiceModel.load()
                }
                model = senseVoiceModel
                tvStatus.text = getString(R.string.status_model_loaded)
                btnRecord.isEnabled = true
                btnSelectFile.isEnabled = true
            } catch (e: Exception) {
                tvStatus.text = getString(R.string.model_not_found)
                Toast.makeText(
                    this@MainActivity,
                    "模型加载失败: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
                btnRecord.isEnabled = false
                btnSelectFile.isEnabled = true
            } finally {
                progressBar.visibility = View.GONE
            }
        }
    }

    private fun toggleRecording() {
        if (isRecording) {
            stopRecordingAndTranscribe()
        } else {
            startRecording()
        }
    }

    private fun startRecording() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Toast.makeText(this, "需要麦克风权限", Toast.LENGTH_SHORT).show()
            requestPermissions()
            return
        }

        recorder = AudioRecorder(this)
        if (recorder?.startRecording() == true) {
            isRecording = true
            tvStatus.text = getString(R.string.status_recording)
            btnRecord.text = getString(R.string.btn_stop)
            btnSelectFile.isEnabled = false
            clearResult()
        } else {
            Toast.makeText(this, "无法启动录音", Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopRecordingAndTranscribe() {
        isRecording = false
        btnRecord.isEnabled = false

        scope.launch {
            tvStatus.text = getString(R.string.status_processing)
            progressBar.visibility = View.VISIBLE

            try {
                val samples = withContext(Dispatchers.IO) {
                    recorder?.stopRecording() ?: FloatArray(0)
                }
                recorder = null

                if (samples.isEmpty()) {
                    tvStatus.text = getString(R.string.status_error)
                    Toast.makeText(this@MainActivity, "录音为空", Toast.LENGTH_SHORT).show()
                    return@launch
                }

                val result = withContext(Dispatchers.Default) {
                    model?.inference(samples)
                }

                if (result != null) {
                    displayResult(result)
                    tvStatus.text = getString(R.string.status_done)
                } else {
                    tvStatus.text = getString(R.string.status_error)
                }
            } catch (e: Exception) {
                tvStatus.text = getString(R.string.status_error)
                Toast.makeText(
                    this@MainActivity,
                    "识别失败: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            } finally {
                progressBar.visibility = View.GONE
                btnRecord.text = getString(R.string.btn_record)
                btnRecord.isEnabled = true
                btnSelectFile.isEnabled = true
            }
        }
    }

    private fun selectAudioFile() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "audio/*"
            addCategory(Intent.CATEGORY_OPENABLE)
        }
        startActivityForResult(
            Intent.createChooser(intent, "选择音频文件"),
            REQUEST_SELECT_FILE
        )
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_SELECT_FILE && resultCode == RESULT_OK) {
            data?.data?.let { uri ->
                transcribeFile(uri)
            }
        }
    }

    private fun transcribeFile(uri: android.net.Uri) {
        tvStatus.text = getString(R.string.status_processing)
        progressBar.visibility = View.VISIBLE
        btnRecord.isEnabled = false
        btnSelectFile.isEnabled = false
        clearResult()

        scope.launch {
            try {
                val samples = withContext(Dispatchers.IO) {
                    val inputStream = contentResolver.openInputStream(uri)
                        ?: throw Exception("无法打开文件")
                    AudioFileReader.readWav(inputStream)
                }

                val result = withContext(Dispatchers.Default) {
                    model?.inference(samples)
                }

                if (result != null) {
                    displayResult(result)
                    tvStatus.text = getString(R.string.status_done)
                } else {
                    tvStatus.text = getString(R.string.status_error)
                }
            } catch (e: Exception) {
                tvStatus.text = getString(R.string.status_error)
                Toast.makeText(
                    this@MainActivity,
                    "识别失败: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            } finally {
                progressBar.visibility = View.GONE
                btnRecord.isEnabled = true
                btnSelectFile.isEnabled = true
            }
        }
    }

    private fun displayResult(result: TokenDecoder.DecodingResult) {
        if (result.language.isNotEmpty()) {
            tvLabelLang.visibility = View.VISIBLE
            tvLanguage.visibility = View.VISIBLE
            tvLanguage.text = result.language
        }
        if (result.emotion.isNotEmpty()) {
            tvLabelEmotion.visibility = View.VISIBLE
            tvEmotion.visibility = View.VISIBLE
            tvEmotion.text = result.emotion
        }
        if (result.event.isNotEmpty()) {
            tvLabelEvent.visibility = View.VISIBLE
            tvEvent.visibility = View.VISIBLE
            tvEvent.text = result.event
        }

        tvLabelText.visibility = View.VISIBLE
        tvResult.text = result.text.ifEmpty { "(空)" }
        tvResult.setTextColor(ContextCompat.getColor(this, R.color.text_primary))
    }

    private fun clearResult() {
        tvResult.text = getString(R.string.hint_result)
        tvResult.setTextColor(ContextCompat.getColor(this, R.color.text_secondary))
        tvLanguage.visibility = View.GONE
        tvEmotion.visibility = View.GONE
        tvEvent.visibility = View.GONE
        tvLabelLang.visibility = View.GONE
        tvLabelEmotion.visibility = View.GONE
        tvLabelEvent.visibility = View.GONE
        tvLabelText.visibility = View.GONE
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isRecording) {
            recorder?.stopRecording()
        }
        model?.release()
    }

    companion object {
        private const val REQUEST_PERMISSIONS = 1001
        private const val REQUEST_SELECT_FILE = 1002
    }
}
