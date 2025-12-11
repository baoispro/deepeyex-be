package subscription

import (
	"time"
)

// PlanName định nghĩa các gói subscription
type PlanName string

const (
	PlanFree       PlanName = "FREE"
	PlanVIP        PlanName = "VIP"
	PlanEnterprise PlanName = "ENTERPRISE"
)

// Subscription model đại diện cho gói subscription của user
type Subscription struct {
	ID           string    `json:"id" gorm:"primaryKey;size:36"`
	UserID       string    `json:"user_id" gorm:"size:36;not null;index"`
	PlanName     PlanName  `json:"plan_name" gorm:"type:varchar(50);not null"`
	AILimit      int       `json:"ai_limit" gorm:"not null;default:0"`      // Số lần quét AI được phép
	ConsultLimit int       `json:"consult_limit" gorm:"not null;default:0"` // Số lần tư vấn được phép
	StartDate    time.Time `json:"start_date" gorm:"not null"`
	EndDate      time.Time `json:"end_date" gorm:"not null"`
	UsedAI       int       `json:"used_ai" gorm:"not null;default:0"`       // Đã dùng bao nhiêu lần AI
	UsedConsult  int       `json:"used_consult" gorm:"not null;default:0"`  // Đã dùng bao nhiêu lần tư vấn
	CreatedAt    time.Time `json:"created_at" gorm:"autoCreateTime"`
	UpdatedAt    time.Time `json:"updated_at" gorm:"autoUpdateTime"`
}

func (Subscription) TableName() string {
	return "subscriptions"
}

// IsValid kiểm tra subscription còn hiệu lực không
func (s *Subscription) IsValid() bool {
	now := time.Now()
	return now.After(s.StartDate) && now.Before(s.EndDate)
}

// CanUseAI kiểm tra có thể dùng AI không
func (s *Subscription) CanUseAI() bool {
	if !s.IsValid() {
		return false
	}
	// -1 = unlimited
	if s.AILimit == -1 {
		return true
	}
	return s.UsedAI < s.AILimit
}

// CanUseConsult kiểm tra có thể dùng tư vấn không
func (s *Subscription) CanUseConsult() bool {
	if !s.IsValid() {
		return false
	}
	// -1 = unlimited
	if s.ConsultLimit == -1 {
		return true
	}
	return s.UsedConsult < s.ConsultLimit
}

// GetPlanLimits trả về limit theo gói
func GetPlanLimits(planName PlanName) (aiLimit, consultLimit int) {
	switch planName {
	case PlanFree:
		return 5, 1 // Free: 5 lần AI, 1 lần tư vấn
	case PlanVIP:
		return -1, 10 // VIP: AI không giới hạn, 10 lần tư vấn (call)
	case PlanEnterprise:
		return -1, -1 // Enterprise: unlimited (-1 = unlimited)
	default:
		return 0, 0
	}
}
