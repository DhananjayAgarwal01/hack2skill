# Implementation Plan: Mental Health Assistant

## Overview

This implementation plan breaks down the Mental Health Assistant into incremental, testable components. The approach follows a bottom-up strategy: building core services first, then integrating them into the orchestration layer, and finally connecting everything through the Streamlit UI. Each major component includes property-based tests to validate correctness properties from the design document.

## Tasks

- [ ] 1. Set up project structure and dependencies
  - Create directory structure: `src/`, `tests/`, `models/`, `config/`
  - Set up Python virtual environment and install core dependencies: streamlit, langchain, transformers, opencv-python, scikit-learn, pandas, numpy, hypothesis (for property testing)
  - Install API client libraries: huggingface_hub, deepgram-sdk, elevenlabs, pygame, sounddevice
  - Create configuration files for API keys and model paths (config.yaml)
  - Set up pytest configuration with hypothesis integration (minimum 100 iterations per property test)
  - _Requirements: 1.5, 2.1, 2.2, 3.1, 5.2, 8.1_

- [ ] 2. Implement Session Manager
  - [ ] 2.1 Create Session and SessionManager classes
    - Implement SessionManager with create_session, get_session, update_session, end_session methods
    - Implement session ID generation using UUID
    - Implement session timeout logic (30 minutes of inactivity)
    - Implement anonymize_session for privacy compliance
    - _Requirements: 10.1, 10.3, 12.4_
  
  - [ ]* 2.2 Write property test for session isolation
    - **Property 31: Session Initialization Isolation**
    - **Property 32: Session Context Isolation**
    - **Validates: Requirements 12.4, 12.5**
  
  - [ ]* 2.3 Write unit tests for session lifecycle
    - Test session creation, retrieval, and timeout behavior
    - Test anonymization removes PII (email, name, IP address)
    - Test session state persistence in Streamlit session_state
    - _Requirements: 10.1, 10.3_

- [ ] 3. Implement Virtual Assistant Service
  - [ ] 3.1 Create VirtualAssistantService with Langchain integration
    - Implement HuggingFace API client for Llama-2-7b-chat-hf model
    - Implement send_message with conversation history using ConversationBufferMemory
    - Implement context compression after 50 exchanges using ConversationSummaryMemory
    - Implement retry logic with exponential backoff (3 attempts, 1s, 2s, 4s delays)
    - Add system prompt to prevent medical diagnosis language
    - Enforce 5-second response timeout
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 12.1, 12.2, 12.3_
  
  - [ ]* 3.2 Write property test for response time constraint
    - **Property 1: Response Time Constraint**
    - **Validates: Requirements 1.2**
  
  - [ ]* 3.3 Write property test for context preservation
    - **Property 2: Context Preservation Across Conversations**
    - **Validates: Requirements 1.3**
  
  - [ ]* 3.4 Write property test for medical diagnosis prohibition
    - **Property 3: Medical Diagnosis Prohibition**
    - **Validates: Requirements 1.4**
  
  - [ ]* 3.5 Write unit tests for Virtual Assistant
    - Test greeting message generation on session start
    - Test conversation history retrieval
    - Test context compression trigger at 50 exchanges
    - Test API retry logic with mocked failures
    - Test system prompt prevents diagnosis language
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 12.3_

- [ ] 4. Implement Voice Interface Service
  - [ ] 4.1 Create VoiceInterfaceService with Deepgram and ElevenLabs integration
    - Implement audio capture using sounddevice library
    - Implement transcribe_audio with Deepgram API client
    - Implement confidence threshold check (0.7) and retry prompt for low confidence
    - Implement synthesize_speech with ElevenLabs API and voice modulation support
    - Implement play_audio using Pygame Mixer with volume control
    - Implement stop_playback functionality
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [ ]* 4.2 Write property test for voice transcription round-trip
    - **Property 4: Voice Transcription Round-Trip**
    - **Validates: Requirements 2.1, 2.4**
  
  - [ ]* 4.3 Write property test for text-to-speech conversion
    - **Property 5: Text-to-Speech Conversion**
    - **Validates: Requirements 2.2**
  
  - [ ]* 4.4 Write unit tests for Voice Interface
    - Test audio capture initialization
    - Test transcription with high/low confidence scores (threshold 0.7)
    - Test speech synthesis with various text lengths
    - Test audio playback and stop functionality
    - Test fallback to text input on transcription failure
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 5. Checkpoint - Core services foundation complete
  - Ensure all tests pass for Session Manager, Virtual Assistant, and Voice Interface
  - Verify API integrations work with test credentials
  - Ask the user if questions arise

- [ ] 6. Implement Emotion Detection Service
  - [ ] 6.1 Create EmotionDetectionService with OpenCV and CNN
    - Load Haar Cascade Classifier for face detection
    - Implement detect_faces using OpenCV with bounding box coordinates
    - Implement extract_features using HOG (Histogram of Oriented Gradients)
    - Implement image preprocessing (48x48 grayscale normalization, pixel values [0, 1])
    - Load pre-trained CNN model for emotion classification (7 classes)
    - Implement classify_emotion with confidence scores
    - Implement process_frame for real-time processing (target: 5+ FPS)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  
  - [ ]* 6.2 Write property test for face detection
    - **Property 6: Face Detection in Frames**
    - **Validates: Requirements 3.1**
  
  - [ ]* 6.3 Write property test for image preprocessing consistency
    - **Property 7: Image Preprocessing Consistency**
    - **Validates: Requirements 3.3**
  
  - [ ]* 6.4 Write property test for emotion classification output format
    - **Property 8: Emotion Classification Output Format**
    - **Validates: Requirements 3.5**
  
  - [ ]* 6.5 Write unit tests for Emotion Detection
    - Test face detection with sample images (with/without faces)
    - Test HOG feature extraction from face regions
    - Test image normalization to 48x48 grayscale with [0, 1] range
    - Test CNN model loading and inference
    - Test frame processing rate meets 5+ FPS requirement
    - Test emotion labels are one of 7 classes (happy, sad, angry, fear, surprise, disgust, neutral)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 7. Implement Questionnaire Engine
  - [ ] 7.1 Create QuestionnaireEngine with configuration loading
    - Load questionnaire structure from JSON/YAML config (PHQ-9, GAD-7, custom questions)
    - Implement get_next_question with progress tracking
    - Implement submit_answer with format validation
    - Implement is_complete check for required questions
    - Implement build_user_profile with PHQ-9 (0-27) and GAD-7 (0-21) scoring
    - Support multiple question types (multiple choice, Likert scale 1-5, free text)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [ ]* 7.2 Write property test for questionnaire sequence consistency
    - **Property 9: Questionnaire Sequence Consistency**
    - **Validates: Requirements 4.1**
  
  - [ ]* 7.3 Write property test for answer validation and optional handling
    - **Property 10: Answer Validation and Optional Handling**
    - **Validates: Requirements 4.2, 4.5**
  
  - [ ]* 7.4 Write property test for UserProfile compilation
    - **Property 11: UserProfile Compilation**
    - **Validates: Requirements 4.3**
  
  - [ ]* 7.5 Write property test for multi-type question support
    - **Property 12: Multi-Type Question Support**
    - **Validates: Requirements 4.4**
  
  - [ ]* 7.6 Write unit tests for Questionnaire Engine
    - Test question loading from config file
    - Test progress tracking through questionnaire
    - Test answer validation for each question type
    - Test PHQ-9 score calculation (0-27 range)
    - Test GAD-7 score calculation (0-21 range)
    - Test optional question skipping doesn't block completion
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 7. Implement Risk Screening Service
  - [ ] 7.1 Create RiskScreeningService with ensemble classifiers
    - Load pre-trained models (Adaboost, GBM, Random Forest, SVM, Decision Tree)
    - Implement feature engineering from UserProfile
    - Implement calculate_risk_score with ensemble aggregation
    - Implement Logistic Regression meta-classifier
    - Implement cosine similarity calculation
    - Implement emotion data incorporation (weighted average)
    - Implement get_risk_factors identification
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  
  - [ ]* 7.2 Write property test for risk score generation
    - **Property 13: Risk Score Generation**
    - **Validates: Requirements 5.1**
  
  - [ ]* 7.3 Write property test for emotion data incorporation
    - **Property 14: Emotion Data Incorporation**
    - **Validates: Requirements 5.5**
  
  - [ ]* 7.4 Write property test for risk score bounds
    - **Property 15: Risk Score Bounds**
    - **Validates: Requirements 5.6**
  
  - [ ]* 7.5 Write unit tests for Risk Screening
    - Test feature engineering from UserProfile
    - Test individual classifier predictions
    - Test ensemble aggregation with Logistic Regression
    - Test cosine similarity calculation
    - Test emotion score weighting (0.3 questionnaire + 0.2 emotion + 0.5 ensemble)
    - Test risk factor identification
    - Test boundary cases (scores at 0.0, 0.5, 1.0)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [ ] 8. Implement Recommendation Engine
  - [ ] 8.1 Create RecommendationEngine with resource database
    - Load resources database (professional help, self-help, crisis hotlines)
    - Implement generate_recommendations with threshold logic (0.7, 0.85)
    - Implement get_crisis_resources
    - Implement get_self_help_plans with risk factor personalization
    - Use non-stigmatizing language templates
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [ ]* 8.2 Write property test for threshold-based recommendation routing
    - **Property 16: Threshold-Based Recommendation Routing**
    - **Validates: Requirements 6.1, 6.2**
  
  - [ ]* 8.3 Write property test for crisis resource inclusion
    - **Property 17: Crisis Resource Inclusion**
    - **Validates: Requirements 6.3**
  
  - [ ]* 8.4 Write property test for risk factor personalization
    - **Property 18: Risk Factor Personalization**
    - **Validates: Requirements 6.4**
  
  - [ ]* 8.5 Write unit tests for Recommendation Engine
    - Test professional help recommendations for high risk (>0.7)
    - Test self-help recommendations for moderate risk (0.4-0.7)
    - Test wellness tips for low risk (<0.4)
    - Test crisis hotline inclusion for critical risk (>0.85)
    - Test personalization based on specific risk factors
    - Test language is non-stigmatizing
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 9. Implement Feedback Collector
  - [ ] 9.1 Create FeedbackCollector with storage integration
    - Implement prompt_feedback UI component
    - Implement submit_feedback with validation
    - Implement feedback anonymization (remove PII)
    - Implement get_feedback_for_training data export
    - Implement run_ab_test for model comparison
    - Support structured (ratings) and unstructured (comments) feedback
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [ ]* 9.2 Write property test for feedback storage with session association
    - **Property 19: Feedback Storage with Session Association**
    - **Validates: Requirements 7.2**
  
  - [ ]* 9.3 Write property test for multi-format feedback support
    - **Property 20: Multi-Format Feedback Support**
    - **Validates: Requirements 7.3**
  
  - [ ]* 9.4 Write property test for feedback anonymization
    - **Property 21: Feedback Anonymization**
    - **Validates: Requirements 7.5**
  
  - [ ]* 9.5 Write unit tests for Feedback Collector
    - Test feedback prompt display after session completion
    - Test structured feedback validation (ratings 1-5)
    - Test unstructured feedback storage (text comments)
    - Test PII detection and removal
    - Test feedback export for training
    - Test A/B test result calculation
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 10. Implement Screening Orchestrator
  - [ ] 10.1 Create ScreeningOrchestrator to coordinate all services
    - Implement initialize_session using SessionManager
    - Implement orchestrate_screening workflow
    - Coordinate Virtual Assistant, Emotion Detector, and Questionnaire Engine
    - Implement data flow: UserProfile → Risk Screener → Recommendation Engine
    - Implement error handling and graceful degradation
    - Implement session completion and cleanup
    - _Requirements: All requirements (orchestration layer)_
  
  - [ ]* 10.2 Write integration tests for screening workflow
    - Test complete screening flow (text-only mode)
    - Test screening with emotion detection enabled
    - Test screening with voice interaction enabled
    - Test graceful degradation when camera unavailable
    - Test graceful degradation when voice services unavailable
    - Test error handling for API failures
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [ ] 11. Implement Model Training Pipeline
  - [ ] 11.1 Create model training scripts
    - Implement train_models for ensemble classifiers
    - Implement tune_hyperparameters using GridSearchCV/RandomizedSearchCV
    - Implement dataset splitting (train/validation/test)
    - Implement model evaluation with precision, recall, F1-score
    - Implement model versioning and artifact storage
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  
  - [ ]* 11.2 Write property test for model evaluation metrics completeness
    - **Property 22: Model Evaluation Metrics Completeness**
    - **Validates: Requirements 8.3**
  
  - [ ]* 11.3 Write property test for dataset separation
    - **Property 23: Dataset Separation**
    - **Validates: Requirements 8.4**
  
  - [ ]* 11.4 Write unit tests for model training
    - Test hyperparameter tuning with sample data
    - Test dataset splitting (no overlap between train/val/test)
    - Test model evaluation metric calculation
    - Test model artifact saving and loading
    - Test A/B testing setup
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 12. Implement Privacy and Security Features
  - [ ] 12.1 Implement privacy compliance features
    - Implement PII detection and anonymization
    - Implement consent management
    - Implement data encryption in transit (TLS)
    - Implement data deletion on request
    - Implement privacy notice display
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  
  - [ ]* 12.2 Write property test for privacy-preserving data storage
    - **Property 24: Privacy-Preserving Data Storage**
    - **Validates: Requirements 10.1, 10.3**
  
  - [ ]* 12.3 Write property test for data deletion completeness
    - **Property 25: Data Deletion Completeness**
    - **Validates: Requirements 10.5**
  
  - [ ]* 12.4 Write unit tests for privacy features
    - Test PII detection (email, name, phone, IP address)
    - Test anonymization of session identifiers
    - Test consent verification before data storage
    - Test data deletion request processing
    - Test privacy notice display
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 13. Implement Error Handling and Resilience
  - [ ] 13.1 Implement comprehensive error handling
    - Implement API retry with exponential backoff (3 attempts)
    - Implement circuit breaker pattern for external APIs
    - Implement graceful degradation for optional services
    - Implement error logging with session context
    - Implement user-friendly error messages
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
  
  - [ ]* 13.2 Write property test for API retry with exponential backoff
    - **Property 26: API Retry with Exponential Backoff**
    - **Validates: Requirements 11.1**
  
  - [ ]* 13.3 Write property test for graceful degradation
    - **Property 27: Graceful Degradation for Optional Services**
    - **Validates: Requirements 11.2, 11.3**
  
  - [ ]* 13.4 Write property test for model failure error handling
    - **Property 28: Model Failure Error Handling**
    - **Validates: Requirements 11.4**
  
  - [ ]* 13.5 Write property test for error logging completeness
    - **Property 29: Error Logging Completeness**
    - **Validates: Requirements 11.5**
  
  - [ ]* 13.6 Write unit tests for error handling
    - Test retry logic with simulated API failures
    - Test circuit breaker state transitions
    - Test graceful degradation paths
    - Test error message formatting
    - Test error logging with PII exclusion
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [ ] 14. Implement Context Management
  - [ ] 14.1 Enhance context management in Virtual Assistant
    - Implement context retrieval for user references
    - Implement context compression at 50+ exchanges
    - Implement session initialization with empty context
    - Implement session isolation (no cross-session context leakage)
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_
  
  - [ ]* 14.2 Write property test for context management and compression
    - **Property 30: Context Management and Compression**
    - **Validates: Requirements 12.2, 12.3**
  
  - [ ]* 14.3 Write property test for session initialization isolation
    - **Property 31: Session Initialization Isolation**
    - **Validates: Requirements 12.4**
  
  - [ ]* 14.4 Write property test for session context isolation
    - **Property 32: Session Context Isolation**
    - **Validates: Requirements 12.5**
  
  - [ ]* 14.5 Write unit tests for context management
    - Test context retrieval for previous message references
    - Test context compression trigger at 50 exchanges
    - Test new session starts with empty context
    - Test concurrent sessions maintain separate contexts
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [ ] 15. Build Streamlit UI
  - [ ] 15.1 Create main Streamlit application
    - Implement home screen with screening options
    - Implement text chat interface
    - Implement voice input/output controls
    - Implement camera widget for emotion detection
    - Implement results dashboard with visualizations
    - Implement responsive layout for desktop/tablet
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  
  - [ ] 15.2 Integrate UI with backend services
    - Connect text input to Virtual Assistant Service
    - Connect voice controls to Voice Interface Service
    - Connect camera to Emotion Detection Service
    - Connect questionnaire display to Questionnaire Engine
    - Connect results display to Risk Screener and Recommendation Engine
    - Connect feedback form to Feedback Collector
    - _Requirements: All requirements (UI integration)_
  
  - [ ]* 15.3 Write UI integration tests
    - Test session initialization from UI
    - Test text message submission and response display
    - Test voice input activation and transcription display
    - Test camera activation and emotion display
    - Test questionnaire navigation and submission
    - Test results dashboard rendering
    - Test feedback form submission
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 16. Create Configuration and Deployment Setup
  - [ ] 16.1 Create configuration files
    - Create config.yaml for API keys and model paths
    - Create questionnaire_config.json for screening questions
    - Create resources_db.json for recommendations database
    - Create logging configuration
    - Create environment variable templates
    - _Requirements: All requirements (configuration)_
  
  - [ ] 16.2 Create deployment documentation
    - Document installation steps
    - Document API key setup (Hugging Face, Deepgram, ElevenLabs)
    - Document model artifact download/setup
    - Document running the application
    - Document testing procedures
    - _Requirements: All requirements (deployment)_

- [ ] 17. End-to-End Testing and Validation
  - [ ] 17.1 Perform comprehensive end-to-end testing
    - Test complete screening flow with all modalities
    - Test edge cases and error scenarios
    - Test performance (response times, frame rates)
    - Test privacy compliance (PII handling, anonymization)
    - Test recommendation accuracy across risk levels
    - _Requirements: All requirements (validation)_
  
  - [ ] 17.2 Conduct user acceptance testing
    - Test with sample users for usability
    - Collect feedback on UI/UX
    - Validate recommendation appropriateness
    - Test accessibility features
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

## Notes

- All property tests should use the `hypothesis` library with minimum 100 iterations
- Each property test should be tagged with the property number and description
- Unit tests should achieve minimum 80% line coverage
- Integration tests should use mocked external APIs for deterministic results
- Model training requires separate dataset preparation (not included in this plan)
- Pre-trained CNN model for emotion detection must be obtained or trained separately
- API keys for Hugging Face, Deepgram, and ElevenLabs must be obtained before testing