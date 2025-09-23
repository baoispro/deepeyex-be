package medicalrecord

type AIRecommendedPlan struct {
	ID           string  `gorm:"primaryKey;size:36" json:"id"`
	AIDiagnosisID string  `gorm:"size:36;not null" json:"ai_diagnosis_id"`
	DrugName     string  `gorm:"size:100" json:"drug_name"`
	Dosage       string  `gorm:"size:50" json:"dosage"`
	Frequency    string  `gorm:"size:50" json:"frequency"`
	DurationDays int  `gorm:"not null" json:"duration_days"`
}
type AddRecommendedPlanRequest struct {
	Description   string `json:"description"`             // Ghi chú thêm, có thể bỏ trống
	DrugName      string `json:"drug_name" binding:"required"` // Tên thuốc
	Dosage        string `json:"dosage" binding:"required"`    // Liều lượng
	Frequency     string `json:"frequency" binding:"required"` // Số lần dùng
	DurationDays  int    `json:"duration_days" binding:"required"` // Số ngày dùng
}