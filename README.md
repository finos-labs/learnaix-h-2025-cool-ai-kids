# 📬 NatWest Hack4aCause hackathon Project Submission Instructions

## Repository Submission Requirements

Each team will be required to submit a GitHub repository with their project. The repository **will** live within the [https://github.com/finos-labs](https://github.com/finos-labs) GitHub Org, and must include the information listed below.

For example, if your team’s name is `strongestavenger`, your repository will be available:
**[`learnaix-h-2025-strongestavenger`](https://github.com/finos-labs/learnaix-h-2025-strongestavenger)**

Please complete this file and include it in the `main` branch of your repository (`README.md`) along with [`HACK4ACAUSE-TEMPLATE_USECASE.docx`](./HACK4ACAUSE-TEMPLATE_USECASE.docx) when submitting your hackathon project.

---

## 📄 Summary of Your Solution (under 150 words)

Briefly describe: This project is a voice-enabled RAG-based Q&A system that lets users upload PDFs and query them through speech.

- What problem does your solution solve?
	- It simplifies exploring long documents by enabling hands-free, interactive question-answering with both text and audio responses—useful for accessibility, research, or business use cases.
- How does it work?
	1. User uploads a PDF → text is extracted, chunked, and indexed in FAISS.
	2. User records a voice query → audio is transcribed to text.
	3. The query is matched against the FAISS index → relevant answer retrieved.
	4. Answer is displayed as text and spoken aloud using TTS.

- What technologies did you use?
	- Gradio for UI.
	- FAISS for semantic search.
	- Speech recognition for transcription.
	- Pyttsx3 for text-to-speech.
	- Custom modules for PDF processing, RAG pipeline, and audio handling.

## 👥 Team Information

| Field            | Details                               |
| ---------------- | ------------------------------------- |
| Team Name        | Cool AI Kids                     	   |
| Title            | LUMO			           |
| Theme            | AI companion			   |
| Contact Email    | arv.arvind@gmail.com         	   |
| Participants     | [Harshita Khandelwal, Reshma Channappanavar, Anurag kedia, Arvind Singh] |
| GitHub Usernames | [@arvind1606, @harshu2908, @enggreshma, @anuragkedia19]   |

---

## 🎥 Submission Video

Provide a video walkthrough/demo of your project. You can upload it to YouTube, Google Drive, Loom, etc.

- 📹 **Video Link**: https://youtu.be/z79zbcbOa4c

---

## 🌐 Hosted App / Solution URL

If your solution is deployed, share the live link here.

- 🌍 **Deployed URL**: https://github.com/finos-labs/learnaix-h-2025-cool-ai-kids.git
---

## License

Copyright 2025 FINOS

Distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

SPDX-License-Identifier: [Apache-2.0](https://spdx.org/licenses/Apache-2.0)
