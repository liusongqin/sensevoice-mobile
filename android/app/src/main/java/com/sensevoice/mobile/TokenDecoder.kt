package com.sensevoice.mobile

/**
 * Decodes CTC output token IDs into text, extracting language, emotion,
 * and event metadata from the SenseVoice model output.
 *
 * SenseVoice output format:
 * - Token 0: Language tag (e.g., <|zh|>, <|en|>)
 * - Token 1: Emotion tag (e.g., <|neutral|>, <|happy|>)
 * - Token 2: Event tag (e.g., <|no_event|>, <|speech|>)
 * - Token 3: ITN tag (e.g., <|withitn|>, <|withoutitn|>)
 * - Token 4+: Actual text content
 */
class TokenDecoder(private val vocabulary: Map<Int, String>) {

    /**
     * Result of decoding SenseVoice output.
     */
    data class DecodingResult(
        val language: String = "",
        val emotion: String = "",
        val event: String = "",
        val text: String = "",
        val rawTokens: List<String> = emptyList()
    )

    /**
     * Performs CTC greedy decoding on logits output from the ONNX model.
     * @param logits Shape [1, numFrames, vocabSize] flattened to 1D
     * @param numFrames Number of time frames
     * @param vocabSize Vocabulary size
     * @param blankId CTC blank token ID (default: 0)
     */
    fun ctcGreedyDecode(
        logits: FloatArray,
        numFrames: Int,
        vocabSize: Int,
        blankId: Int = 0
    ): DecodingResult {
        val tokenIds = mutableListOf<Int>()
        var prevTokenId = blankId

        for (t in 0 until numFrames) {
            val offset = t * vocabSize
            var maxId = 0
            var maxVal = logits[offset]
            for (v in 1 until vocabSize) {
                if (logits[offset + v] > maxVal) {
                    maxVal = logits[offset + v]
                    maxId = v
                }
            }
            if (maxId != blankId && maxId != prevTokenId) {
                tokenIds.add(maxId)
            }
            prevTokenId = maxId
        }

        return decodeTokenIds(tokenIds)
    }

    /**
     * Decodes a list of token IDs into a DecodingResult.
     */
    fun decodeTokenIds(tokenIds: List<Int>): DecodingResult {
        if (tokenIds.isEmpty()) {
            return DecodingResult()
        }

        val tokens = tokenIds.map { id -> vocabulary[id] ?: "<unk>" }

        var language = ""
        var emotion = ""
        var event = ""
        val textTokens = mutableListOf<String>()

        for ((index, token) in tokens.withIndex()) {
            when {
                index == 0 && isLanguageTag(token) -> {
                    language = extractTagContent(token)
                }
                index == 1 && isEmotionTag(token) -> {
                    emotion = extractTagContent(token)
                }
                index == 2 && isEventTag(token) -> {
                    event = extractTagContent(token)
                }
                index == 3 && isItnTag(token) -> {
                    // Skip ITN tag
                }
                !isSpecialTag(token) -> {
                    textTokens.add(token)
                }
            }
        }

        val text = mergeTokensToText(textTokens)

        return DecodingResult(
            language = language,
            emotion = emotion,
            event = event,
            text = text,
            rawTokens = tokens
        )
    }

    private fun mergeTokensToText(tokens: List<String>): String {
        val sb = StringBuilder()
        for (token in tokens) {
            val cleaned = token.replace("▁", " ").replace("Ġ", " ")
            sb.append(cleaned)
        }
        return sb.toString().trim()
    }

    private fun isLanguageTag(token: String): Boolean {
        return token.matches(Regex("<\\|[a-z]+\\|>"))
    }

    private fun isEmotionTag(token: String): Boolean {
        return token in setOf(
            "<|neutral|>", "<|happy|>", "<|sad|>", "<|angry|>",
            "<|fearful|>", "<|disgusted|>", "<|surprised|>"
        )
    }

    private fun isEventTag(token: String): Boolean {
        return token in setOf(
            "<|no_event|>", "<|speech|>", "<|music|>", "<|noise|>",
            "<|applause|>", "<|laughter|>"
        )
    }

    private fun isItnTag(token: String): Boolean {
        return token in setOf("<|withitn|>", "<|withoutitn|>")
    }

    private fun isSpecialTag(token: String): Boolean {
        return token.startsWith("<|") && token.endsWith("|>")
    }

    private fun extractTagContent(tag: String): String {
        return tag.removePrefix("<|").removeSuffix("|>")
    }

    companion object {
        /**
         * Loads vocabulary from tokens.txt file content.
         * Each line is formatted as: "token id"
         */
        fun loadVocabulary(content: String): Map<Int, String> {
            val vocab = mutableMapOf<Int, String>()
            content.lines().forEach { line ->
                val trimmed = line.trim()
                if (trimmed.isNotEmpty()) {
                    val lastSpace = trimmed.lastIndexOf(' ')
                    if (lastSpace > 0) {
                        val token = trimmed.substring(0, lastSpace)
                        val id = trimmed.substring(lastSpace + 1).trim().toIntOrNull()
                        if (id != null) {
                            vocab[id] = token
                        }
                    }
                }
            }
            return vocab
        }
    }
}
