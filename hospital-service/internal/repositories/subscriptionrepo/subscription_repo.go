package subscriptionrepo

import (
	"errors"
	"hospital-service/internal/models/subscription"
	"time"

	"gorm.io/gorm"
)

type SubscriptionRepo struct {
	db *gorm.DB
}

func NewSubscriptionRepo(db *gorm.DB) *SubscriptionRepo {
	return &SubscriptionRepo{db: db}
}

// Create tạo subscription mới
func (r *SubscriptionRepo) Create(sub *subscription.Subscription) error {
	return r.db.Create(sub).Error
}

// GetByUserID lấy subscription hiện tại của user
func (r *SubscriptionRepo) GetByUserID(userID string) (*subscription.Subscription, error) {
	var sub subscription.Subscription
	err := r.db.Where("user_id = ?", userID).
		Order("created_at DESC").
		First(&sub).Error
	
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil // Không có subscription
		}
		return nil, err
	}
	
	return &sub, nil
}

// GetActiveByUserID lấy subscription đang active của user
func (r *SubscriptionRepo) GetActiveByUserID(userID string) (*subscription.Subscription, error) {
	var sub subscription.Subscription
	now := time.Now()
	
	err := r.db.Where("user_id = ? AND start_date <= ? AND end_date >= ?", userID, now, now).
		Order("created_at DESC").
		First(&sub).Error
	
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}
		return nil, err
	}
	
	return &sub, nil
}

// Update cập nhật subscription
func (r *SubscriptionRepo) Update(sub *subscription.Subscription) error {
	return r.db.Save(sub).Error
}

// DeleteOldSubscription xóa subscription cũ của user (trước khi tạo mới)
func (r *SubscriptionRepo) DeleteOldSubscription(userID string) error {
	return r.db.Where("user_id = ?", userID).Delete(&subscription.Subscription{}).Error
}

// IncrementUsage tăng số lần đã dùng
func (r *SubscriptionRepo) IncrementUsage(userID string, isAI bool) error {
	sub, err := r.GetActiveByUserID(userID)
	if err != nil {
		return err
	}
	if sub == nil {
		return errors.New("no active subscription found")
	}

	if isAI {
		sub.UsedAI++
	} else {
		sub.UsedConsult++
	}

	return r.Update(sub)
}
