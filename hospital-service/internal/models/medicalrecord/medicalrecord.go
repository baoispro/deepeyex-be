package medicalrecord

import "time"

type MedicalRecord struct {
	RecordID        string         `gorm:"primaryKey;size:36" json:"record_id"`
	PatientID       string         `gorm:"size:36;not null" json:"patient_id"`
	AppointmentID   string         `gorm:"size:36" json:"appointment_id"`
	DoctorID        string         `gorm:"size:36;" json:"doctor_id"`
	Diagnosis       string         `gorm:"type:text" json:"diagnosis"`
	Notes           *string        `gorm:"type:text" json:"notes,omitempty"`
	RelatedRecordID *string        `gorm:"size:36" json:"related_record_id"` // nếu là tái khám
	CreatedAt       time.Time      `gorm:"autoCreateTime"`
	UpdatedAt       time.Time      `gorm:"autoUpdateTime"`
	AI_Diagnoses    []AIDiagnosis  `gorm:"foreignKey:RecordID;references:RecordID" json:"ai_diagnoses"`
	Attachments     []Attachment   `gorm:"foreignKey:RecordID;references:RecordID" json:"attachments"`
	Prescriptions   []Prescription `gorm:"foreignKey:MedicalRecordID;references:RecordID" json:"prescriptions"`
}

// Request khi gọi endpoint
type InitRecordAndDiagnosisRequest struct {
	PatientID    string  `json:"patient_id" binding:"required" example:"abc123"`
	DiseaseCode  string  `json:"disease_code" binding:"required" example:"D001"`
	Confidence   float64 `json:"confidence" binding:"required" example:"0.89"`
	Diagnosis    string  `json:"diagnosis" example:"AI preliminary diagnosis"`
	MainImageURL string  `json:"main_image_url" binding:"required" example:"https://s3.../image.jpg"` // URL hình ảnh chẩn đoán
	EyeType      *string `json:"eye_type,omitempty" example:"RIGHT"`                                  // "LEFT" | "RIGHT" | "BOTH"
	Notes        *string `json:"notes,omitempty" example:"Patient complained about blurry vision"`    // Ghi chú thêm
}

// Response trả về cả Record và Diagnosis
type InitRecordAndDiagnosisResponse struct {
	RecordID  string      `json:"record_id"`
	PatientID string      `json:"patient_id"`
	CreatedAt time.Time   `json:"created_at"`
	Diagnosis AIDiagnosis `json:"ai_diagnosis"`
}
