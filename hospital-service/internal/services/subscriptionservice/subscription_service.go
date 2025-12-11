package subscriptionservice

import (
	"errors"
	"hospital-service/internal/models/subscription"
	"hospital-service/internal/repositories/subscriptionrepo"
	"hospital-service/internal/services/paymentservice"
	"time"

	"github.com/google/uuid"
)

type SubscriptionService struct {
	repo          *subscriptionrepo.SubscriptionRepo
	paymentService *paymentservice.VnpayService
}

func NewSubscriptionService(repo *subscriptionrepo.SubscriptionRepo, paymentService *paymentservice.VnpayService) *SubscriptionService {
	return &SubscriptionService{
		repo:          repo,
		paymentService: paymentService,
	}
}

// SubscribeRequest request để subscribe gói
type SubscribeRequest struct {
	UserID   string `json:"user_id" binding:"required"`
	PlanName string `json:"plan_name" binding:"required"` // FREE, VIP, ENTERPRISE
	Duration int    `json:"duration"`                     // Số ngày (mặc định 30)
}

// SubscribeResponse response sau khi subscribe
type SubscribeResponse struct {
	ID           string    `json:"id"`
	UserID       string    `json:"user_id"`
	PlanName     string    `json:"plan_name"`
	AILimit      int       `json:"ai_limit"`
	ConsultLimit int       `json:"consult_limit"`
	StartDate    time.Time `json:"start_date"`
	EndDate      time.Time `json:"end_date"`
	UsedAI       int       `json:"used_ai"`
	UsedConsult  int       `json:"used_consult"`
}

// CheckLimitResponse response khi check limit
type CheckLimitResponse struct {
	CanUse      bool `json:"can_use"`
	Remaining   int  `json:"remaining"`   // Số lần còn lại
	Used        int  `json:"used"`        // Đã dùng
	Limit       int  `json:"limit"`       // Tổng limit
	IsUnlimited bool `json:"is_unlimited"` // Có phải unlimited không
}

// SubscribeRequestPayment response cho gói cần thanh toán
type SubscribeRequestPayment struct {
	PaymentURL     string `json:"payment_url"`
	Amount         int    `json:"amount"`
	PlanName       string `json:"plan_name"`
	Duration       int    `json:"duration"`
	UserID         string `json:"user_id"`
	SubscriptionID string `json:"subscription_id"` // ID để frontend gửi lại khi complete payment
}

// GetPlanPrice trả về giá gói theo VND
func GetPlanPrice(planName subscription.PlanName, duration int) int {
	switch planName {
	case subscription.PlanFree:
		return 0 // Miễn phí
	case subscription.PlanVIP:
		return 299000 * duration / 30 // 200k/tháng, tính theo số ngày
	case subscription.PlanEnterprise:
		return 500000 * duration / 30 // 500k/tháng, tính theo số ngày
	default:
		return 0
	}
}

// Subscribe đăng ký gói subscription mới
// - FREE: tạo subscription ngay
// - VIP/ENTERPRISE: tạo payment URL để thanh toán
func (s *SubscriptionService) Subscribe(req SubscribeRequest) (*SubscribeResponse, *SubscribeRequestPayment, error) {
	// Validate plan name
	planName := subscription.PlanName(req.PlanName)
	if planName != subscription.PlanFree && planName != subscription.PlanVIP && planName != subscription.PlanEnterprise {
		return nil, nil, errors.New("invalid plan name. Must be FREE, VIP, or ENTERPRISE")
	}

	// Set duration mặc định 30 ngày nếu không có
	duration := req.Duration
	if duration <= 0 {
		duration = 30
	}

	// Kiểm tra subscription hiện tại
	currentSub, err := s.repo.GetActiveByUserID(req.UserID)
	if err != nil {
		return nil, nil, err
	}

	// Nếu đang có subscription đang active
	if currentSub != nil && currentSub.IsValid() {
		currentPlan := currentSub.PlanName

		// Rule 1: Nếu đang có FREE, chỉ không cho đăng ký FREE mới, nhưng vẫn cho đăng ký VIP/ENTERPRISE
		if currentPlan == subscription.PlanFree && planName == subscription.PlanFree {
			return nil, nil, errors.New("bạn đang có gói FREE đang hoạt động, vui lòng đợi hết hạn trước khi đăng ký gói FREE mới")
		}

		// Rule 2: Nếu đang có VIP/ENTERPRISE, không cho đăng ký FREE
		if (currentPlan == subscription.PlanVIP || currentPlan == subscription.PlanEnterprise) && planName == subscription.PlanFree {
			return nil, nil, errors.New("bạn đang có gói " + string(currentPlan) + " đang hoạt động, không thể đăng ký gói FREE")
		}
	}

	// Nếu là FREE, tạo subscription ngay
	if planName == subscription.PlanFree {
		// Xóa subscription cũ
		if err := s.repo.DeleteOldSubscription(req.UserID); err != nil {
			return nil, nil, err
		}

		// Lấy limits theo gói
		aiLimit, consultLimit := subscription.GetPlanLimits(planName)

		// Tạo subscription mới
		startDate := time.Now()
		endDate := startDate.AddDate(0, 0, duration)

		newSub := &subscription.Subscription{
			ID:           uuid.NewString(),
			UserID:       req.UserID,
			PlanName:     planName,
			AILimit:      aiLimit,
			ConsultLimit: consultLimit,
			StartDate:    startDate,
			EndDate:      endDate,
			UsedAI:       0,
			UsedConsult:  0,
		}

		if err := s.repo.Create(newSub); err != nil {
			return nil, nil, err
		}

		return &SubscribeResponse{
			ID:           newSub.ID,
			UserID:       newSub.UserID,
			PlanName:     string(newSub.PlanName),
			AILimit:      newSub.AILimit,
			ConsultLimit: newSub.ConsultLimit,
			StartDate:    newSub.StartDate,
			EndDate:      newSub.EndDate,
			UsedAI:       newSub.UsedAI,
			UsedConsult:  newSub.UsedConsult,
		}, nil, nil
	}

	// VIP/ENTERPRISE: Tạo payment URL
	amount := GetPlanPrice(planName, duration)
	subscriptionID := uuid.NewString()

	// Tạo payment URL với subscription ID và thông tin
	paymentURL, err := s.paymentService.CreatePaymentURLForSubscription(amount, subscriptionID, req.UserID, req.PlanName, duration)
	if err != nil {
		return nil, nil, err
	}

	return nil, &SubscribeRequestPayment{
		PaymentURL:     paymentURL,
		Amount:         amount,
		PlanName:       req.PlanName,
		Duration:       duration,
		UserID:         req.UserID,
		SubscriptionID: subscriptionID,
	}, nil
}

// CompleteSubscription tạo subscription sau khi thanh toán thành công
func (s *SubscriptionService) CompleteSubscription(subscriptionID, userID, planName string, duration int) (*SubscribeResponse, error) {
	plan := subscription.PlanName(planName)
	if plan != subscription.PlanVIP && plan != subscription.PlanEnterprise {
		return nil, errors.New("invalid plan for payment completion")
	}

	// Xóa subscription cũ
	if err := s.repo.DeleteOldSubscription(userID); err != nil {
		return nil, err
	}

	// Lấy limits theo gói
	aiLimit, consultLimit := subscription.GetPlanLimits(plan)

	// Tạo subscription mới
	startDate := time.Now()
	endDate := startDate.AddDate(0, 0, duration)

	newSub := &subscription.Subscription{
		ID:           subscriptionID,
		UserID:       userID,
		PlanName:     plan,
		AILimit:      aiLimit,
		ConsultLimit: consultLimit,
		StartDate:    startDate,
		EndDate:      endDate,
		UsedAI:       0,
		UsedConsult:  0,
	}

	if err := s.repo.Create(newSub); err != nil {
		return nil, err
	}

	return &SubscribeResponse{
		ID:           newSub.ID,
		UserID:       newSub.UserID,
		PlanName:     string(newSub.PlanName),
		AILimit:      newSub.AILimit,
		ConsultLimit: newSub.ConsultLimit,
		StartDate:    newSub.StartDate,
		EndDate:      newSub.EndDate,
		UsedAI:       newSub.UsedAI,
		UsedConsult:  newSub.UsedConsult,
	}, nil
}

// CheckAILimit kiểm tra có thể dùng AI không
func (s *SubscriptionService) CheckAILimit(userID string) (*CheckLimitResponse, error) {
	sub, err := s.repo.GetActiveByUserID(userID)
	if err != nil {
		return nil, err
	}

	if sub == nil {
		// Không có subscription, tạo free plan mặc định
		req := SubscribeRequest{
			UserID:   userID,
			PlanName: string(subscription.PlanFree),
			Duration: 30,
		}
		_, _, err := s.Subscribe(req)
		if err != nil {
			return nil, err
		}
		// Lấy lại sau khi tạo
		sub, err = s.repo.GetActiveByUserID(userID)
		if err != nil {
			return nil, err
		}
		if sub == nil {
			return nil, errors.New("failed to create default subscription")
		}
	}

	isUnlimited := sub.AILimit == -1
	canUse := sub.IsValid() && (isUnlimited || sub.UsedAI < sub.AILimit)
	remaining := -1 // unlimited
	if !isUnlimited {
		remaining = sub.AILimit - sub.UsedAI
		if remaining < 0 {
			remaining = 0
		}
	}

	return &CheckLimitResponse{
		CanUse:      canUse,
		Remaining:   remaining,
		Used:        sub.UsedAI,
		Limit:       sub.AILimit,
		IsUnlimited: isUnlimited,
	}, nil
}

// CheckConsultLimit kiểm tra có thể dùng tư vấn không
func (s *SubscriptionService) CheckConsultLimit(userID string) (*CheckLimitResponse, error) {
	sub, err := s.repo.GetActiveByUserID(userID)
	if err != nil {
		return nil, err
	}

	if sub == nil {
		// Không có subscription, tạo free plan mặc định
		req := SubscribeRequest{
			UserID:   userID,
			PlanName: string(subscription.PlanFree),
			Duration: 30,
		}
		_, _, err := s.Subscribe(req)
		if err != nil {
			return nil, err
		}
		// Lấy lại sau khi tạo
		sub, err = s.repo.GetActiveByUserID(userID)
		if err != nil {
			return nil, err
		}
		if sub == nil {
			return nil, errors.New("failed to create default subscription")
		}
	}

	isUnlimited := sub.ConsultLimit == -1
	canUse := sub.IsValid() && (isUnlimited || sub.UsedConsult < sub.ConsultLimit)
	remaining := -1 // unlimited
	if !isUnlimited {
		remaining = sub.ConsultLimit - sub.UsedConsult
		if remaining < 0 {
			remaining = 0
		}
	}

	return &CheckLimitResponse{
		CanUse:      canUse,
		Remaining:   remaining,
		Used:        sub.UsedConsult,
		Limit:       sub.ConsultLimit,
		IsUnlimited: isUnlimited,
	}, nil
}

// UpdateUsage cập nhật số lần đã dùng
func (s *SubscriptionService) UpdateUsage(userID string, isAI bool) error {
	return s.repo.IncrementUsage(userID, isAI)
}

// GetSubscription lấy subscription hiện tại của user
func (s *SubscriptionService) GetSubscription(userID string) (*SubscribeResponse, error) {
	sub, err := s.repo.GetActiveByUserID(userID)
	if err != nil {
		return nil, err
	}

	if sub == nil {
		return nil, nil // Không có subscription
	}

	return &SubscribeResponse{
		ID:           sub.ID,
		UserID:       sub.UserID,
		PlanName:     string(sub.PlanName),
		AILimit:      sub.AILimit,
		ConsultLimit: sub.ConsultLimit,
		StartDate:    sub.StartDate,
		EndDate:      sub.EndDate,
		UsedAI:       sub.UsedAI,
		UsedConsult:  sub.UsedConsult,
	}, nil
}
