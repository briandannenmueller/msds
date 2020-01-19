# CSC464 Computer Audition

## Homework

### Homework 1: Basic audio preprocessing with Matlab

Practicing basic audio preprocessing in Matlab: resample, segmentation, spectrogram, MIDI-conversion,  additive-synthesis, etc.

Get familiar with Audacity.



### Homework 2: Pitch Detection

Implement the auto-correlation method: YIN algorithm and test it with clear/noised/multi-pitched audio pieces.

Discussion of polyphonic music piece pitch detection, designed a way to connect the pitches over time into pitch contours so that each contour corresponds to one instrument(not implemented).



### Homework 3: Onset Detection & Tempo Estimation & Beat Tracking by Dynamic Programming

#### Part 1: Onset Detection

- Implemented an energy-based onset detection function, in which onsets are chosen by thresholding and peak-picking.
- Implemented a spectral-based onset detection function. Again, onsets are again chosen through thresholding and peak-picking. There is a problem that many small peaks that are not corresponding to onsets are also strong in onset strength curve. Those peaks are removed by subtracting the local mean.
- Compared the two methods.

#### Part 2: Tempogram and tempo estimation

Perform STFT on the onset strength curve to get tempogram, pay attention to the axis.We can tell the fundamental frequency i.e.tempo by inspecting the tempogram.

#### Part 3: Beat tracking by dynamic programing

 Implement the dynamic-programming-based beat tracking approach by [Ellis](http://www2.ece.rochester.edu/~zduan/teaching/ece477/reading/Ellis_BeatTrackingByDynamicProgramming_07.pdf).



### Homework 4: Non-negative Matrix Factorization(NMF) Source Separation & Viterbi algorithm

#### Part 1: NMF source separation

Implemented the NMF using K-L divergence and the multiplicative rule. Test the NMF implementation on a source separation task.

#### Part 2: Viterbi algorithm

 Implemented the Viterbi algorithm to smooth the single pitch detection results.



### Homework 5: **Singing voice separation with neural networks**

This homework to about applying neural networks to a traditional signal processing task: **separating singing voice from single-channel mixture**. Includes two NN architectures: feed-forward NN and LSTM. Use python3 and pytorch.



## Final Project

ATTENTION BASED 3-D CONVOLUTION NEURAL NETWORK FOR SPEECH EMOTION RECOGNITION

Abstract:Speech Emotion Recognition is important in many fields like human-machine interaction and health care. Recently, various deep neural networks are introduced to this do- main and achieved great performance. Yet SER remains a difficult task due to several obstacles, e.g. the sparsity of emotion-relevant frames in a long speech utterance, the difficulties of bettering extracted acoustic feature quality, and the long-dependencies problem in RNN models. In this paper, we proposed a 3-D Convolutional Neural Net- work with attention mechanism, which helps to generate higher quality spectrogram features and mitigate the spar- sity issue. The experimental results on the popular dataset IEMOCAP show the proposed approach outperforms the baseline models.

Paper and poster inside the folder.

Also includes peer reviews from three other students, and for other three students.



## Topic Presentation

Our chosen topic is: **speech reverberation**



## Paper Reviews

In this course, we are required to review 7 papers corresponding to the final project we choose. Every review includes:

1. Title and author of the paper

2. Summary of the paper. (a couple of sentences)

   This should take only several sentences and should be in your own language instead of

   copying the paper’s abstract. It demonstrates that you understand the paper.

3. Good things about the paper. (one paragraph)

   There must be something good you feel about the paper. It can be some good ideas, interesting problems, and clear illustrations. It can also be something you learned from the paper, or inspirations for follow up work.

4. Major comments (several paragraphs)
    Discuss the weaknesses of the assumptions, formulation, technical approach, explanation, presentation, reasoning, experimental setup, results, analysis, literature review, comparison with other work, etc. Give some suggestions to improve. Be constructive.

5. Minor comments (a list)
    Comment on other weaknesses including typos, grammatical errors, figures, etc. that do not significantly affect the overall quality of the paper.

Here is the list of paper I choose:

1. YIN, a fundamental frequency estimation for speech and music, by: Alain de Cheveigne**, **Hideki Kawahara
2. End**-**to**-**End Model for Speech Enhancement by Consistent Spectrogram Masking, by: Du**, **Xingjian**, **Zhu**, **Mengyao**, **Shi**, **Xuan**, **Zhang**, **Xinpeng**, **Zhang**, **Wen**, **Chen**, **Jingdong
3. DBLSTM-Based Multi-scale Fusion for Dynamic Emotion Prediction in Music, by:Xinxing Li, Jiashen Tian, Mingxing Xu, Yishuang Ning, Lianhong Cai
4. Stacked Convolutional and Recurrent Neural Networks for Music Emotion Recognition, by:Miroslav Malik**, **Sharath Adavanne**, **Konstantinos Drossos**, **Tuomas Virtanen**, **Dasa Ticha**, **and Roman Jarina
5. End-to-End Speech Emotion Recognition Using Deep Neural Networks, by: Panagiotis Tzirakis, Jiehao Zhang, Bjorn W. Schuller
6. Effective Attention Mechanism in Dynamic Models for Speech Emotion Recognition, by: Po-Wei Hsiao and Chia-Ping Chen
7. Self-supervised audio representation learning for mobile devices, by:Marco Tagliasacchi, Beat Gfeller, Félix de Chaumont Quitry, Dominik Roblek













