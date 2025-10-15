package medicalrecord

import "time"

type AIDiagnosis struct {
	ID           string    `gorm:"primaryKey;size:36" json:"id"`
	PatientID    string    `gorm:"size:36;not null" json:"patient_id"`
	RecordID     string    `gorm:"size:36;not null" json:"record_id"`
	DiseaseCode  string    `gorm:"size:50;not null" json:"disease_code"`
	Confidence   float64   `gorm:"type:decimal(5,4);not null" json:"confidence"`
	MainImageURL string    `gorm:"size:500" json:"main_image_url"`    // URL hình ảnh chính dùng để chẩn đoán
	EyeType      *string   `gorm:"size:10" json:"eye_type,omitempty"` // "LEFT" | "RIGHT" | "BOTH"
	Notes        *string   `gorm:"type:text" json:"notes,omitempty"`  // Ghi chú thêm
	CreatedAt    time.Time `gorm:"autoCreateTime" json:"created_at"`

	VerifiedBy      *string    `gorm:"size:36" json:"verified_by,omitempty"`        // ID bác sĩ xác nhận
	VerifiedAt      *time.Time `json:"verified_at,omitempty"`                       // thời điểm bác sĩ xác nhận
	VerificationSig *string    `gorm:"type:text" json:"verification_sig,omitempty"` // chữ ký điện tử (base64 hoặc URL)
	Status          string     `gorm:"size:20;default:'PENDING'" json:"status"`     // PENDING, APPROVED, REJECTED
	DoctorNotes     *string    `gorm:"type:text" json:"doctor_notes,omitempty"`     // ghi chú bác sĩ khi xác nhận
}

func (AIDiagnosis) TableName() string {
	return "ai_diagnoses"
}
