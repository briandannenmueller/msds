####Speech Dereverberation

Nov.25, 2019



####Concepts:

Distortion

 - Reverberation
   	- Convolution of the direct sound
   	- The room impulse response (RIR)
	- Background noise

#### Applications

Speech Enhancement

Automatic Speech Recognition(ASR)

More specific:(**IBM slides, 2008**)

Automotive: Hands-Free Car Phone Kits. Health Hearing Aids, Home-Care. 

Home/Office: Speech and Speaker: Recognition, Internet Telephony, Teleconferencing, Set-top boxes, Home Automation. 

Mobile: Mobile Phones, Smartphones, PDA’s, Mobile Multimedia Systems. 

#### Methods

**Classical non-learning methods**

- Inverse filtering
  - Theory: magnitude relationship between anechoic signal and reverberant version is consistent
  - Pros: has supporting theory
  - Cons: 
    - Inverse filter cannot get directly and estimation is hard(don't really understand why it is hard)
    - Some conditions can not be satisfies in practice(**Neely and Ellen, 1979**)
  - History:
    - **Miyoshi,1988; Taylor, 2010(Speech dereverberation book ); Mingyang Wu, 2006**
- Other acoustic methods
  - Temporal envelop filtering: **Avendano, 1996(2)**
  - Spectral subtraction: **Lebart, 2001(21)**
  - Pitch-based: **Roman, 2006(30)**

**Learning methods**

- Ideal Binary Mask(IBM): 

  - learn the spectral mapping from reverberant speech to its anechoic signal. with the IBM as the goal, the  problem becomes a binary classification problem

  - GMM-based: **Akula, 2018; Ma, 2018**

  - DNNs:

    - Pros:

      - Automatically learn more abstract features as the number of layers increases
      - More abstract features tend to be more invariant to superficial variations 

    - Cons:

      - Hard to train 
      - No temporal information

    - History:

      Perceptron: **Jin, 2007**

      Pre train using restricted Boltzmann machines and only reverberant: **Han, 2014**

      4-layer DNN: **Han, 2015**(required paper)

      Room-aware NN and mtilti-task learning: **Giri, 2015** 

      5-layer DNN: **Zhao, 2015**(required paper)

      WPE+CNN: **Escudero, 2018**

      2-stage 2-layer DNN: **Zhao, 2019**

  - RNNs

    - Content-based RNN: **Santos, 2017**
    - LSTM: **Zhao, 2018**
    - BLSTM, de-nosing and de-reverberation:  Weninger, 2014

- Other:
  -  Velmurugan, and P. Rao, “A Non-convolutive NMF Model for Speech Dereverberation,” in Interspeech 2018, 2018, pp. 1324–1328.
  -  reverberation time (RT) estimation: Weninger, 2014



