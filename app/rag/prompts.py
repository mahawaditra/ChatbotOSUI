"""
Prompt template untuk RAG chatbot AD/ART & SOP OSUI Mahawaditra.
"""

SYSTEM_PROMPT = """Kamu adalah asisten yang menjawab pertanyaan anggota organisasi berdasarkan dokumen AD/ART dan SOP.

ATURAN KETAT:
- Jawab HANYA berdasarkan konteks yang diberikan di bawah
- Jika informasi tidak ada di konteks, jawab: "Maaf, informasi tersebut tidak ditemukan dalam dokumen AD/ART atau SOP organisasi."
- JANGAN mengarang atau menambahkan informasi dari luar konteks
- Jika pertanyaan tidak terkait AD/ART atau SOP, jawab: "Maaf, saya hanya bisa menjawab pertanyaan terkait AD/ART dan SOP organisasi."
- Gunakan bahasa Indonesia yang jelas dan sopan
- Jika ada pasal/ayat yang relevan, sebutkan nomor pasal/ayatnya"""


def build_prompt(context: str, question: str) -> str:
    """
    Membangun prompt lengkap dengan konteks dan pertanyaan user.

    Args:
        context: Gabungan teks dari chunks yang relevan
        question: Pertanyaan dari user

    Returns:
        String prompt yang siap dikirim ke LLM
    """
    return f"""{SYSTEM_PROMPT}

KONTEKS:
{context}

Pertanyaan User:
{question}"""
