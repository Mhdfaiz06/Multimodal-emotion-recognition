# Multimodal Emotion Recognition System

This project bridges the gap between research-oriented AI models and scalable software engineering. It is a production-grade, hybrid Multimodal Emotion Recognition system designed to analyze both *what* is being said (textual semantics) and *how* it is being said (acoustic prosody). 

By fusing state-of-the-art transformer models with a robust backend architecture, the system achieves high-accuracy inference while remaining easily deployable.

### Core Architecture
*   **Acoustic Processing:** Leverages **wav2vec2** to extract prosodic features—such as pitch and intensity—from raw audio signals.
*   **Linguistic Processing:** Utilizes **RoBERTa** for deep semantic analysis of the corresponding text.
*   **Fusion Mechanism:** Implements **cross-modal attention** to intelligently weigh and combine auditory and textual signals for the final emotional classification.
*   **Data Pipeline:** Built around the **TESS** (Toronto Emotional Speech Set) dataset, utilizing a structured data manifest to cleanly align audio and text pairs.

### Deployment
The inference pipeline is served via a **FastAPI** backend for low-latency requests and is fully containerized using **Docker**, ensuring a consistent, environment-agnostic setup out of the box.
