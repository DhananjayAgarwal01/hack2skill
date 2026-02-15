# Requirements Document

## Introduction

The Mental Health Assistant is a comprehensive screening application that combines conversational AI, computer vision-based emotion detection, and machine learning classifiers to assess mental health risk levels. The system provides an accessible, privacy-conscious interface for users to receive preliminary mental health screening through multiple interaction modalities (text, voice, and visual), with appropriate recommendations based on risk assessment scores.

## Glossary

- **Virtual_Assistant**: The conversational AI component powered by Llama-2-7b-chat-hf that interacts with users
- **Emotion_Detector**: The computer vision module that analyzes facial expressions to detect emotional states
- **Risk_Screener**: The ML classifier ensemble that calculates mental health risk scores
- **Voice_Interface**: The audio processing system combining speech-to-text and text-to-speech capabilities
- **Questionnaire_Engine**: The component that manages and presents screening questions to users
- **Recommendation_Engine**: The system that provides appropriate next steps based on risk scores
- **Feedback_Collector**: The module that gathers user feedback for model improvement
- **Risk_Score**: A numerical value (0-1) representing the assessed mental health risk level
- **Risk_Threshold**: The cutoff value (typically 0.7) above which professional help is recommended
- **Session**: A single user interaction from start to completion of screening
- **User_Profile**: The aggregated data from questionnaire responses and emotion detection

## Requirements

### Requirement 1: Virtual Assistant Interaction

**User Story:** As a user, I want to interact with a conversational AI assistant, so that I can receive mental health screening in a comfortable, natural dialogue format.

#### Acceptance Criteria

1. WHEN a user initiates a session, THE Virtual_Assistant SHALL greet the user and explain the screening process
2. WHEN a user sends a text message, THE Virtual_Assistant SHALL generate a contextually appropriate response within 5 seconds
3. WHEN a conversation exceeds 10 exchanges, THE Virtual_Assistant SHALL maintain context from previous messages using conversation history
4. WHEN a user asks clarifying questions about the screening, THE Virtual_Assistant SHALL provide informative responses without medical diagnosis
5. THE Virtual_Assistant SHALL use the Llama-2-7b-chat-hf model via Hugging Face API for response generation

### Requirement 2: Voice Interaction Capabilities

**User Story:** As a user, I want to interact using voice input and receive spoken responses, so that I can complete screening hands-free or when typing is inconvenient.

#### Acceptance Criteria

1. WHEN a user activates voice input, THE Voice_Interface SHALL capture audio and convert it to text using Deepgram API
2. WHEN the Virtual_Assistant generates a response, THE Voice_Interface SHALL convert text to speech using ElevenLabs API
3. WHEN audio playback is initiated, THE Voice_Interface SHALL use Pygame Mixer to play the synthesized speech
4. WHEN voice recognition fails or produces unclear input, THE Voice_Interface SHALL prompt the user to repeat or switch to text input
5. THE Voice_Interface SHALL support voice modulation for natural-sounding responses

### Requirement 3: Real-Time Emotion Detection

**User Story:** As a user, I want the system to detect my emotional state through my facial expressions, so that the screening can incorporate non-verbal emotional cues.

#### Acceptance Criteria

1. WHEN a user enables camera access, THE Emotion_Detector SHALL detect faces using Haar Cascade Classifiers in real-time
2. WHEN a face is detected, THE Emotion_Detector SHALL extract facial features using HOG (Histogram of Oriented Gradients)
3. WHEN facial features are extracted, THE Emotion_Detector SHALL normalize and convert the image to grayscale before classification
4. WHEN the preprocessed image is ready, THE Emotion_Detector SHALL classify emotions using a trained CNN model
5. WHEN emotion classification completes, THE Emotion_Detector SHALL output emotion labels with confidence scores
6. THE Emotion_Detector SHALL process frames at minimum 5 FPS for responsive feedback

### Requirement 4: Mental Health Questionnaire

**User Story:** As a user, I want to answer structured screening questions, so that the system can assess my mental health status comprehensively.

#### Acceptance Criteria

1. WHEN a user begins the questionnaire, THE Questionnaire_Engine SHALL present questions in a logical sequence
2. WHEN a user submits an answer, THE Questionnaire_Engine SHALL validate the response format before proceeding
3. WHEN all required questions are answered, THE Questionnaire_Engine SHALL compile responses into a User_Profile
4. THE Questionnaire_Engine SHALL support multiple question types (multiple choice, Likert scale, free text)
5. WHEN a user skips optional questions, THE Questionnaire_Engine SHALL continue without requiring those responses

### Requirement 5: Risk Assessment and Scoring

**User Story:** As a user, I want to receive an accurate mental health risk assessment, so that I can understand whether I should seek professional help.

#### Acceptance Criteria

1. WHEN a User_Profile is complete, THE Risk_Screener SHALL calculate a Risk_Score using ensemble ML classifiers
2. THE Risk_Screener SHALL use Adaboost, Gradient Boosting Machines, Random Forest, SVM, and Decision Trees for classification
3. WHEN multiple classifier outputs are available, THE Risk_Screener SHALL aggregate predictions using Logistic Regression
4. WHEN calculating risk similarity, THE Risk_Screener SHALL use cosine similarity to match user responses against known risk profiles
5. WHEN emotion detection data is available, THE Risk_Screener SHALL incorporate emotion scores into the final Risk_Score calculation
6. THE Risk_Screener SHALL output a Risk_Score between 0.0 and 1.0

### Requirement 6: Recommendation Generation

**User Story:** As a user, I want to receive appropriate recommendations based on my screening results, so that I know what steps to take next.

#### Acceptance Criteria

1. WHEN a Risk_Score exceeds the Risk_Threshold, THE Recommendation_Engine SHALL suggest professional mental health resources
2. WHEN a Risk_Score is below the Risk_Threshold, THE Recommendation_Engine SHALL provide self-help plans and coping strategies
3. WHEN generating recommendations, THE Recommendation_Engine SHALL include crisis hotline information if Risk_Score exceeds 0.85
4. THE Recommendation_Engine SHALL provide personalized recommendations based on specific risk factors identified
5. WHEN displaying recommendations, THE Recommendation_Engine SHALL present information in clear, non-stigmatizing language

### Requirement 7: User Feedback Collection

**User Story:** As a system administrator, I want to collect user feedback on screening accuracy, so that the models can be continuously improved.

#### Acceptance Criteria

1. WHEN a screening session completes, THE Feedback_Collector SHALL prompt users to provide feedback on their experience
2. WHEN a user submits feedback, THE Feedback_Collector SHALL store the feedback with associated Session metadata
3. THE Feedback_Collector SHALL support structured feedback (ratings) and unstructured feedback (text comments)
4. WHEN sufficient feedback is collected, THE Feedback_Collector SHALL enable A/B testing for feedback classification models
5. THE Feedback_Collector SHALL anonymize all feedback data before storage

### Requirement 8: Model Training and Optimization

**User Story:** As a data scientist, I want to optimize model hyperparameters and retrain classifiers, so that screening accuracy improves over time.

#### Acceptance Criteria

1. WHEN training ensemble classifiers, THE Risk_Screener SHALL use Grid Search CV or Randomized Search CV for hyperparameter tuning
2. WHEN new feedback data is available, THE Risk_Screener SHALL support retraining of classification models
3. WHEN evaluating model performance, THE Risk_Screener SHALL calculate precision, recall, and F1-score metrics
4. THE Risk_Screener SHALL maintain separate training, validation, and test datasets
5. WHEN a new model version is trained, THE Risk_Screener SHALL perform A/B testing before deployment

### Requirement 9: User Interface and Dashboard

**User Story:** As a user, I want an intuitive web interface to access all screening features, so that I can easily navigate the application.

#### Acceptance Criteria

1. THE System SHALL provide a Streamlit-based web application interface
2. WHEN a user accesses the application, THE System SHALL display a clear home screen with screening options
3. THE System SHALL provide separate interface sections for text chat, voice interaction, and emotion detection
4. WHEN displaying results, THE System SHALL present Risk_Score and recommendations in an easy-to-understand dashboard
5. THE System SHALL support responsive design for desktop and tablet devices

### Requirement 10: Privacy and Data Security

**User Story:** As a user, I want my personal health information to be protected, so that I can trust the system with sensitive data.

#### Acceptance Criteria

1. THE System SHALL not store personally identifiable information without explicit user consent
2. WHEN processing user data, THE System SHALL encrypt all data in transit using TLS
3. WHEN storing session data, THE System SHALL anonymize user identifiers
4. THE System SHALL provide clear privacy notices before data collection
5. WHEN a user requests data deletion, THE System SHALL remove all associated session data within 24 hours

### Requirement 11: Error Handling and Graceful Degradation

**User Story:** As a user, I want the system to handle errors gracefully, so that technical issues don't prevent me from completing screening.

#### Acceptance Criteria

1. WHEN an API call fails (Hugging Face, Deepgram, or ElevenLabs), THE System SHALL retry up to 3 times with exponential backoff
2. WHEN the camera is unavailable, THE System SHALL continue screening without emotion detection
3. WHEN voice services are unavailable, THE System SHALL fall back to text-only interaction
4. WHEN the ML model fails to generate a Risk_Score, THE System SHALL notify the user and suggest retrying or seeking direct professional help
5. IF any critical component fails, THEN THE System SHALL log the error with sufficient detail for debugging

### Requirement 12: Conversational Context Management

**User Story:** As a user, I want the assistant to remember our conversation, so that I don't have to repeat information.

#### Acceptance Criteria

1. THE Virtual_Assistant SHALL use Langchain to manage conversational context across multiple exchanges
2. WHEN a user references previous statements, THE Virtual_Assistant SHALL retrieve relevant context from conversation history
3. WHEN a session exceeds 50 exchanges, THE Virtual_Assistant SHALL summarize and compress older context to maintain performance
4. WHEN a new session starts, THE Virtual_Assistant SHALL initialize with empty context
5. THE Virtual_Assistant SHALL maintain context for the duration of a single screening session only
