package appointmentrepo

import (
	"hospital-service/internal/models/appointment"
	"time"

	"gorm.io/gorm"
)

type TimeSlotRepo struct {
	db *gorm.DB
}

func NewTimeSlotRepo(db *gorm.DB) *TimeSlotRepo {
	return &TimeSlotRepo{db: db}
}

func (r *TimeSlotRepo) Create(slot *appointment.TimeSlot) error {
	return r.db.Create(slot).Error
}

func (r *TimeSlotRepo) FindByID(id string) (*appointment.TimeSlot, error) {
	var slot appointment.TimeSlot
	if err := r.db.First(&slot, "slot_id = ?", id).Error; err != nil {
		return nil, err
	}
	return &slot, nil
}

func (r *TimeSlotRepo) FindByDoctorID(doctorID string) ([]appointment.TimeSlot, error) {
	var slots []appointment.TimeSlot
	if err := r.db.Where("doctor_id = ?", doctorID).Find(&slots).Error; err != nil {
		return nil, err
	}
	return slots, nil
}

func (r *TimeSlotRepo) Update(slot *appointment.TimeSlot) error {
	return r.db.Save(slot).Error
}

func (r *TimeSlotRepo) Delete(id string) error {
	return r.db.Delete(&appointment.TimeSlot{}, "slot_id = ?", id).Error
}

func (r *TimeSlotRepo) ListAll() ([]appointment.TimeSlot, error) {
	var slots []appointment.TimeSlot
	if err := r.db.Preload("Doctor").Find(&slots).Error; err != nil {
		return nil, err
	}
	return slots, nil
}

// Check xem có slot trùng thời gian cho doctor
func (r *TimeSlotRepo) CountOverlapping(doctorID string, startTime, endTime time.Time) (int64, error) {
	var count int64
	err := r.db.Model(&appointment.TimeSlot{}).
		Where("doctor_id = ? AND NOT (end_time <= ? OR start_time >= ?)", doctorID, startTime, endTime).
		Count(&count).Error
	return count, err
}