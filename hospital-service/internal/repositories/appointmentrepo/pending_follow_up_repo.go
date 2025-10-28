package appointmentrepo

import (
	"hospital-service/internal/models/appointment"
	"time"

	"gorm.io/gorm"
)

type PendingFollowUpRepo struct {
	db *gorm.DB
}

func NewPendingFollowUpRepo(db *gorm.DB) *PendingFollowUpRepo {
	return &PendingFollowUpRepo{db: db}
}

// Create tạo mới pending follow-up appointment
func (r *PendingFollowUpRepo) Create(p *appointment.PendingFollowUpAppointment) error {
	return r.db.Create(p).Error
}

// GetByToken lấy pending appointment theo confirmation token
func (r *PendingFollowUpRepo) GetByToken(token string) (*appointment.PendingFollowUpAppointment, error) {
	var p appointment.PendingFollowUpAppointment
	err := r.db.Where("confirmation_token = ?", token).First(&p).Error
	if err != nil {
		return nil, err
	}
	return &p, nil
}

// GetByID lấy pending appointment theo ID
func (r *PendingFollowUpRepo) GetByID(id string) (*appointment.PendingFollowUpAppointment, error) {
	var p appointment.PendingFollowUpAppointment
	err := r.db.Where("pending_id = ?", id).First(&p).Error
	if err != nil {
		return nil, err
	}
	return &p, nil
}

// Update cập nhật pending appointment
func (r *PendingFollowUpRepo) Update(p *appointment.PendingFollowUpAppointment) error {
	return r.db.Save(p).Error
}

// Delete xóa pending appointment
func (r *PendingFollowUpRepo) Delete(id string) error {
	return r.db.Where("pending_id = ?", id).Delete(&appointment.PendingFollowUpAppointment{}).Error
}

// DeleteExpired xóa các pending appointment đã hết hạn
func (r *PendingFollowUpRepo) DeleteExpired() error {
	now := time.Now()
	return r.db.Where("expires_at < ?", now).Delete(&appointment.PendingFollowUpAppointment{}).Error
}

