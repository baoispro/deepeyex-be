package medicalrecord

import "time"

type Prescription struct {
	PrescriptionID  string             `gorm:"primaryKey;size:36" json:"prescription_id"`
	AI_DiagnosisID  *string            `gorm:"size:36" json:"ai_diagnosis_id,omitempty"`
	MedicalRecordID *string            `gorm:"size:36" json:"medical_record_id,omitempty"`
	PatientID       string             `gorm:"size:36;not null" json:"patient_id"`
	Source          string             `gorm:"size:20" json:"source"` // AI hoặc DOCTOR
	Description     *string            `gorm:"type:text" json:"description,omitempty"`
	Status          string             `gorm:"size:20;default:'PENDING'" json:"status"` // PENDING, APPROVED, REJECTED
	Items           []PrescriptionItem `gorm:"foreignKey:PrescriptionID;references:PrescriptionID" json:"items,omitempty"`
	CreatedAt       time.Time          `gorm:"autoCreateTime"`
	UpdatedAt       time.Time          `gorm:"autoUpdateTime"`
}

type PrescriptionItem struct {
	ItemID         string `gorm:"primaryKey;size:36" json:"prescription_item_id"`
	PrescriptionID string `gorm:"size:36;not null" json:"prescription_id"`
	DrugName       string `gorm:"size:100" json:"drug_name"`
	Dosage         string `gorm:"size:50" json:"dosage"`
	Frequency      string `gorm:"size:50" json:"frequency"`
	DurationDays   int    `gorm:"not null" json:"duration_days"`
	Notes          *string   `gorm:"type:text" json:"notes,omitempty"`
    StartDate      time.Time `json:"start_date"`
    EndDate        time.Time `json:"end_date"` // auto tính = StartDate + DurationDays
}
