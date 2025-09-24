package medicalrecord

import "time"

type MedicalRecord struct {
	RecordID       string     `gorm:"primaryKey;size:36" json:"record_id"`
	PatientID      string     `gorm:"size:36;not null" json:"patient_id"`
	AppointmentID  string     `gorm:"size:36" json:"appointment_id"`
	DoctorID       string     `gorm:"size:36;" json:"doctor_id"`
	Diagnosis      string     `gorm:"type:text" json:"diagnosis"`
	CreatedBy      string     `gorm:"size:20" json:"created_by"` // "AI" | "DOCTOR"
	CreatedAt      time.Time  `gorm:"autoCreateTime"`
	UpdatedAt      time.Time  `gorm:"autoUpdateTime"`
	AI_Diagnoses   []AIDiagnosis `gorm:"foreignKey:RecordID;references:RecordID" json:"ai_diagnoses"`
	Attachments    []Attachment  `gorm:"foreignKey:RecordID;references:RecordID" json:"attachments"`
	FollowUps      []FollowUp    `gorm:"foreignKey:RecordID;references:RecordID" json:"follow_ups"`
}

// Request khi gọi endpoint
type InitRecordAndDiagnosisRequest struct {
	PatientID    string  `json:"patient_id" binding:"required" example:"abc123"`
	DiseaseCode  string  `json:"disease_code" binding:"required" example:"D001"`
	Confidence   float64 `json:"confidence" binding:"required" example:"0.89"`
	Diagnosis    string  `json:"diagnosis" example:"AI preliminary diagnosis"`
}

// Response trả về cả Record và Diagnosis
type InitRecordAndDiagnosisResponse struct {
	RecordID   string    `json:"record_id"`
	PatientID  string    `json:"patient_id"`
	CreatedAt  time.Time `json:"created_at"`
	Diagnosis  AIDiagnosis `json:"ai_diagnosis"`
}