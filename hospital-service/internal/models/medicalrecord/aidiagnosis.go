package medicalrecord

import "time"

type AIDiagnosis struct {
	ID               string              `gorm:"primaryKey;size:36" json:"id"`
	RecordID         string              `gorm:"size:36;not null" json:"record_id"`
	DiseaseCode      string              `gorm:"size:50;not null" json:"disease_code"`
	Confidence       float64             `gorm:"type:decimal(5,4);not null" json:"confidence"`
	MainImageURL     string              `gorm:"size:500" json:"main_image_url"`                       // URL hình ảnh chính dùng để chẩn đoán
	EyeType          *string             `gorm:"size:10" json:"eye_type,omitempty"`                    // "LEFT" | "RIGHT" | "BOTH"
	Notes            *string             `gorm:"type:text" json:"notes,omitempty"`                     // Ghi chú thêm
	RecommendedPlans []AIRecommendedPlan `gorm:"foreignKey:AIDiagnosisID;references:ID;constraint:OnUpdate:CASCADE,OnDelete:CASCADE" json:"recommended_plans,omitempty"`
	CreatedAt        time.Time           `gorm:"autoCreateTime" json:"created_at"`
}

type CreateAIDiagnosisRequest struct {
	RecordID     string  `json:"record_id" binding:"required"`
	DiseaseCode  string  `json:"disease_code" binding:"required"`
	Confidence   float64 `json:"confidence" binding:"required"`
	MainImageURL string  `json:"main_image_url" binding:"required"` // URL hình ảnh chẩn đoán
	EyeType      *string `json:"eye_type,omitempty"`                // "LEFT" | "RIGHT" | "BOTH"
	Notes        *string `json:"notes,omitempty"`                   // Ghi chú
}

func (AIDiagnosis) TableName() string {
	return "ai_diagnoses"
}
