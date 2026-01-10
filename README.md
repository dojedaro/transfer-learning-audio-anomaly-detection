🔊 Transfer Learning for Audio Anomaly Detection

🎛️ Machine Sound Classification with ToyADMOS-Style Data

📌 Overview

This project demonstrates an end-to-end transfer learning pipeline for audio anomaly detection using machine sound signals.
Raw audio waveforms are converted into mel-spectrogram images, which are then classified using convolutional neural networks (CNNs).

The core focus is to empirically evaluate the impact of transfer learning across different machine domains.

✅ Training from scratch
✅ Training with transfer learning
✅ Quantitative comparison of both approaches

The results show strong positive transfer, significantly improving anomaly detection performance.

🎯 Key Contributions

✅ Synthetic generation of ToyADMOS-style machine audio
✅ Conversion of audio signals into mel-spectrogram images
✅ Careful dataset splitting (train / validation / test)
✅ CNN training without transfer learning (baseline)
✅ Cross-machine transfer learning and fine-tuning
✅ Metric-driven evaluation (Accuracy, F1 Score)
✅ Clear demonstration of positive vs. negative transfer

🗂️ Dataset Description

Synthetic audio is generated to simulate the ToyADMOS dataset structure, enabling controlled experimentation.

🏭 Machine Types

• ToyCar
• ToyTrain
• ToyConveyor

🎧 Audio Properties

• Sampling rate: 16 kHz
• Duration: 10 seconds per sample
• Normal + anomalous operating sounds
• Frequency distortions and noise perturbations for anomalies

Both raw audio (.wav) and mel-spectrogram images (.png) are produced.

🧪 Experimental Pipeline
1️⃣ Audio Generation

Machine-specific signals are synthesized with injected anomalies.

2️⃣ Feature Extraction

Audio is converted into 128-bin mel-spectrograms, saved as images.

3️⃣ Dataset Preparation

Two setups are used:
• Random balanced split
• Machine-specific split for transfer learning

4️⃣ Baseline Model (No Transfer Learning)

CNN trained from scratch using:
• ToyCar + ToyTrain → training
• ToyConveyor → testing

5️⃣ Transfer Learning

The baseline CNN is reused and fine-tuned on ToyConveyor data only.

6️⃣ Evaluation

Models are evaluated using:
• Accuracy
• F1 Score
• Training time
• Inference time

🧠 Model Architecture
🔹 Baseline CNN

• 4 convolutional blocks
• Batch normalization + max pooling
• Fully connected layers with dropout
• ~19M trainable parameters

🔹 Transfer Learning Model

• Initialized from baseline CNN
• Fine-tuned with a reduced learning rate
• Faster convergence and improved generalization

📊 Results
📈 Performance on ToyConveyor (Test Set)

❌ Without Transfer Learning
• Accuracy: 0.50
• F1 Score: 0.00

✅ With Transfer Learning
• Accuracy: 0.94
• F1 Score: 0.94

🚀 Transfer learning delivers a dramatic improvement, confirming strong cross-machine knowledge transfer.
