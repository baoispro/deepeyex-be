package hospital

import (
	"time"
)

// Hospital đại diện cho thông tin bệnh viện
type Hospital struct {
	HospitalID string `json:"hospital_id" gorm:"type:uuid;primaryKey"` // PK
	Name       string    `json:"name" gorm:"size:100;not null"`
	Address    string    `json:"address" gorm:"size:255"`
	Phone      string    `json:"phone" gorm:"size:20"`
	Email      string    `json:"email" gorm:"size:100"`
	
	CreatedAt  time.Time `json:"created_at" gorm:"autoCreateTime"`
	UpdatedAt time.Time `json:"updated_at"`

}

