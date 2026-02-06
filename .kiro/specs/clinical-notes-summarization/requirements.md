# Requirements Document

## Introduction

The Clinical Notes Summarization & Risk Flagging System is an AI-powered Flask application that automatically processes clinical notes to generate concise summaries and identify potential health risks. The system uses NLP techniques (rule-based and ML approaches) to support healthcare workflow efficiency while maintaining strict ethical boundaries around synthetic/de-identified data usage only.

## Glossary

- **System**: The Clinical Notes Summarization & Risk Flagging System
- **Clinical_Note**: A text document containing patient medical information
- **Preprocessor**: The text preprocessing engine component
- **Summarizer**: The summarization module that generates concise summaries
- **Risk_Engine**: The risk detection engine that identifies health risks
- **Dashboard**: The frontend interface for displaying results
- **Risk_Category**: Classification of health risks (Cardiac, Diabetes, Respiratory, Medication_Compliance)
- **Synthetic_Data**: Artificially generated clinical notes that do not contain real patient information
- **De_Identified_Data**: Real clinical notes with all personally identifiable information removed
- **ROUGE_Score**: Recall-Oriented Understudy for Gisting Evaluation metric for summarization quality
- **Confidence_Score**: A numerical value indicating the certainty of a risk prediction

## Requirements

### Requirement 1: Clinical Note Upload and Storage

**User Story:** As a healthcare professional, I want to upload clinical notes to the system, so that I can receive automated summaries and risk assessments.

#### Acceptance Criteria

1. WHEN a user uploads a clinical note via the web interface, THE System SHALL accept text files and plain text input
2. WHEN a clinical note is received, THE System SHALL store the raw note in the /data/raw_notes/ directory
3. WHEN storing a clinical note, THE System SHALL assign a unique identifier to each note
4. WHEN a clinical note exceeds 100KB in size, THE System SHALL reject the upload and return an error message
5. THE System SHALL only accept UTF-8 encoded text files

### Requirement 2: Text Preprocessing

**User Story:** As a system component, I want to preprocess clinical notes, so that downstream NLP tasks receive clean, normalized text.

#### Acceptance Criteria

1. WHEN a raw clinical note is processed, THE Preprocessor SHALL convert all text to lowercase
2. WHEN preprocessing text, THE Preprocessor SHALL tokenize the text into individual words and sentences
3. WHEN tokenizing, THE Preprocessor SHALL remove common English stopwords
4. WHEN processing tokens, THE Preprocessor SHALL apply lemmatization to reduce words to their base forms
5. WHEN encountering medical terminology, THE Preprocessor SHALL normalize medical terms to standard forms
6. WHEN preprocessing is complete, THE Preprocessor SHALL store cleaned text in /data/cleaned_notes/
7. WHEN preprocessing fails, THE Preprocessor SHALL log the error and return a descriptive error message

### Requirement 3: Extractive Summarization

**User Story:** As a healthcare professional, I want to receive concise summaries of lengthy clinical notes, so that I can quickly understand patient status without reading full documents.

#### Acceptance Criteria

1. WHEN a cleaned clinical note is provided, THE Summarizer SHALL generate a summary containing 5-7 sentences
2. WHEN generating summaries, THE Summarizer SHALL use extractive NLP techniques (TF-IDF or TextRank)
3. WHEN selecting sentences, THE Summarizer SHALL prioritize sentences with high importance scores
4. WHEN a clinical note contains fewer than 5 sentences, THE Summarizer SHALL return the original text as the summary
5. WHEN summarization is complete, THE Summarizer SHALL calculate a compression ratio (summary length / original length)
6. THE Summarizer SHALL preserve the original sentence order in the summary

### Requirement 4: Risk Detection and Flagging

**User Story:** As a healthcare professional, I want the system to automatically identify potential health risks in clinical notes, so that I can prioritize patient care and interventions.

#### Acceptance Criteria

1. WHEN analyzing a clinical note, THE Risk_Engine SHALL identify risk indicators for Cardiac, Diabetes, Respiratory, and Medication_Compliance categories
2. WHEN detecting risks using rule-based methods, THE Risk_Engine SHALL match predefined medical trigger patterns (blood pressure values, smoking mentions, glucose levels)
3. WHEN using ML classification, THE Risk_Engine SHALL apply a trained Logistic Regression or Random Forest model
4. WHEN a risk is detected, THE Risk_Engine SHALL assign a confidence score between 0.0 and 1.0
5. WHEN multiple risks are detected, THE Risk_Engine SHALL return all identified risks with their respective categories and confidence scores
6. WHEN no risks are detected, THE Risk_Engine SHALL return an empty risk list

### Requirement 5: Flask API Endpoints

**User Story:** As a frontend application, I want to interact with the backend through RESTful API endpoints, so that I can orchestrate the processing pipeline.

#### Acceptance Criteria

1. THE System SHALL provide a POST endpoint at /upload_note that accepts clinical note text
2. THE System SHALL provide a POST endpoint at /process_note that triggers the full processing pipeline
3. THE System SHALL provide a GET endpoint at /get_summary that returns the generated summary for a given note ID
4. THE System SHALL provide a GET endpoint at /get_risks that returns detected risks for a given note ID
5. WHEN an API endpoint receives a request, THE System SHALL return JSON-formatted responses
6. WHEN an API request fails, THE System SHALL return appropriate HTTP status codes (400 for bad requests, 500 for server errors)
7. WHEN processing is successful, THE System SHALL return HTTP status code 200

### Requirement 6: Result Dashboard Display

**User Story:** As a healthcare professional, I want to view summarized reports and risk alerts in a clear dashboard, so that I can quickly assess patient information.

#### Acceptance Criteria

1. WHEN a user accesses the dashboard, THE Dashboard SHALL display an interface for uploading clinical text
2. WHEN a summary is generated, THE Dashboard SHALL display the summary text prominently
3. WHEN risks are detected, THE Dashboard SHALL highlight risk alerts with visual indicators (colors or icons)
4. WHEN displaying risks, THE Dashboard SHALL show the risk category, description, and confidence score
5. WHEN displaying any results, THE Dashboard SHALL include a disclaimer stating the system is for educational and workflow support only
6. THE Dashboard SHALL clearly state that the system is not a diagnostic tool

### Requirement 7: Synthetic and De-Identified Data Handling

**User Story:** As a system administrator, I want to ensure the system only processes synthetic or de-identified data, so that patient privacy is protected and ethical guidelines are followed.

#### Acceptance Criteria

1. THE System SHALL only accept synthetic clinical notes or de-identified public datasets
2. WHEN data is loaded, THE System SHALL verify that no personally identifiable information (PII) is present
3. THE System SHALL include documentation stating it is designed for synthetic/de-identified data only
4. THE System SHALL not store or process real patient information
5. WHEN displaying results, THE System SHALL include warnings about data limitations

### Requirement 8: Model Evaluation and Metrics

**User Story:** As a system developer, I want to evaluate the performance of summarization and risk detection models, so that I can ensure quality and identify areas for improvement.

#### Acceptance Criteria

1. WHEN evaluating summarization quality, THE System SHALL calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
2. WHEN evaluating summarization, THE System SHALL calculate the compression ratio
3. WHEN evaluating risk detection, THE System SHALL calculate precision and recall metrics
4. WHEN evaluating risk classification, THE System SHALL generate a confusion matrix
5. THE System SHALL provide an evaluation script that can be run on test datasets
6. WHEN evaluation is complete, THE System SHALL output metrics in a structured format (JSON or CSV)

### Requirement 9: Error Handling and Logging

**User Story:** As a system administrator, I want comprehensive error handling and logging, so that I can troubleshoot issues and maintain system reliability.

#### Acceptance Criteria

1. WHEN any component encounters an error, THE System SHALL log the error with timestamp, component name, and error details
2. WHEN preprocessing fails, THE System SHALL return a descriptive error message to the user
3. WHEN summarization fails, THE System SHALL return a fallback response indicating the issue
4. WHEN risk detection fails, THE System SHALL continue processing and return available results
5. THE System SHALL maintain log files in a /logs/ directory
6. WHEN critical errors occur, THE System SHALL return HTTP 500 status codes with error descriptions

### Requirement 10: Model Training and Persistence

**User Story:** As a system developer, I want to train and persist ML models for risk detection, so that the system can make predictions without retraining.

#### Acceptance Criteria

1. THE System SHALL provide a training script for the risk classification model
2. WHEN training is complete, THE System SHALL serialize the trained model to /risk_engine/ml_model.pkl
3. WHEN the application starts, THE System SHALL load the persisted model from disk
4. WHEN the model file is missing, THE System SHALL log an error and fall back to rule-based detection only
5. THE System SHALL support retraining the model with new synthetic datasets
6. WHEN saving models, THE System SHALL include metadata (training date, accuracy metrics, feature list)

### Requirement 11: Responsible AI Compliance

**User Story:** As a healthcare organization, I want the system to comply with responsible AI principles, so that it is used ethically and safely.

#### Acceptance Criteria

1. THE System SHALL display a prominent disclaimer that it is not a diagnostic tool
2. THE System SHALL state that it is designed for educational and workflow support purposes only
3. WHEN displaying risk predictions, THE System SHALL include confidence scores to indicate uncertainty
4. THE System SHALL document all limitations in user-facing documentation
5. THE System SHALL not make treatment recommendations
6. THE System SHALL include clear statements about the use of synthetic/de-identified data only
