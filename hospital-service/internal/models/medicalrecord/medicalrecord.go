package medicalrecord

import (
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/models/doctor"
	"time"
)

type MedicalRecord struct {
	RecordID        string                  `gorm:"primaryKey;size:36" json:"record_id"`
	PatientID       string                  `gorm:"size:36;not null" json:"patient_id"`
	AppointmentID   string                  `gorm:"size:36" json:"appointment_id"`
	DoctorID        string                  `gorm:"size:36;" json:"doctor_id"`
	Diagnosis       string                  `gorm:"type:text" json:"diagnosis"`
	Notes           *string                 `gorm:"type:text" json:"notes,omitempty"`
	RelatedRecordID *string                 `gorm:"size:36" json:"related_record_id"` // nếu là tái khám
	CreatedAt       time.Time               `gorm:"autoCreateTime"`
	UpdatedAt       time.Time               `gorm:"autoUpdateTime"`
	AI_Diagnoses    []AIDiagnosis           `gorm:"foreignKey:RecordID;references:RecordID" json:"ai_diagnoses"`
	Attachments     []Attachment            `gorm:"foreignKey:RecordID;references:RecordID" json:"attachments"`
	Prescriptions   []Prescription          `gorm:"foreignKey:MedicalRecordID;references:RecordID" json:"prescriptions"`
	Appointment     appointment.Appointment `gorm:"foreignKey:AppointmentID;references:AppointmentID" json:"appointment,omitempty"`
	Doctor          doctor.Doctor          `gorm:"foreignKey:DoctorID;references:DoctorID" json:"doctor,omitempty"`
}

// Request khi gọi endpoint
type InitRecordAndDiagnosisRequest struct {
	PatientID     string `json:"patient_id" binding:"required" example:"abc123"`
	AppointmentID string `json:"appointment_id,omitempty" example:"appt456"` // có thể null
	DoctorID      string `json:"doctor_id,omitempty" example:"doc789"`       // có thể null
	AIDiagnosisID string `json:"ai_diagnosis_id,omitempty" example:"diag101"` // có thể null
}

// Response trả về cả Record và Diagnosis
type InitRecordAndDiagnosisResponse struct {
	RecordID  string      `json:"record_id"`
}
