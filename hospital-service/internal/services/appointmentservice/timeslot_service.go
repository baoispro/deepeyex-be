package appointmentservice

import (
	"errors"
	"fmt"
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/repositories/appointmentrepo"
	"time"

	"github.com/google/uuid"
)

type TimeSlotService struct {
	repo *appointmentrepo.TimeSlotRepo
}

func NewTimeSlotService(repo *appointmentrepo.TimeSlotRepo) *TimeSlotService {
	return &TimeSlotService{repo: repo}
}

// ---------------- Create ----------------
func (s *TimeSlotService) Create(doctorID string, startTime, endTime time.Time, capacity int) (*appointment.TimeSlot, error) {
	if doctorID == "" {
		return nil, errors.New("doctor ID is required")
	}
	if capacity <= 0 {
		return nil, errors.New("capacity must be greater than 0")
	}
	if !startTime.Before(endTime) {
		return nil, errors.New("startTime must be before endTime")
	}

	// Kiểm tra trùng slot
	count, err := s.repo.CountOverlapping(doctorID, startTime, endTime)
	if err != nil {
		return nil, fmt.Errorf("failed to check overlapping slots: %v", err)
	}
	if count > 0 {
		return nil, errors.New("doctor already has a timeslot in this time range")
	}

	slot := &appointment.TimeSlot{
		SlotID:    generateSlotID(),
		DoctorID:  doctorID,
		StartTime: startTime,
		EndTime:   endTime,
		Capacity:  capacity,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if err := s.repo.Create(slot); err != nil {
		return nil, err
	}

	return slot, nil
}

func (s *TimeSlotService) GetByID(id string) (*appointment.TimeSlot, error) {
	return s.repo.FindByID(id)
}

func (s *TimeSlotService) GetByDoctorID(doctorID string) ([]appointment.TimeSlot, error) {
	return s.repo.FindByDoctorID(doctorID)
}

// ---------------- Update ----------------
func (s *TimeSlotService) Update(slotID string, startTime, endTime *time.Time, capacity *int) (*appointment.TimeSlot, error) {
	// Lấy slot hiện tại
	slot, err := s.repo.FindByID(slotID)
	if err != nil {
		return nil, fmt.Errorf("timeslot not found: %v", err)
	}

	// Update fields nếu có
	if startTime != nil {
		slot.StartTime = *startTime
	}
	if endTime != nil {
		slot.EndTime = *endTime
	}
	if capacity != nil {
		if *capacity <= 0 {
			return nil, errors.New("capacity must be greater than 0")
		}
		slot.Capacity = *capacity
	}

	if !slot.StartTime.Before(slot.EndTime) {
		return nil, errors.New("startTime must be before endTime")
	}

	count, err := s.repo.CountOverlapping(slot.DoctorID, slot.StartTime, slot.EndTime)
	if err != nil {
		return nil, fmt.Errorf("failed to check overlapping slots: %v", err)
	}
	if count > 0 {
		return nil, errors.New("doctor already has a timeslot in this time range")
	}

	slot.UpdatedAt = time.Now()
	if err := s.repo.Update(slot); err != nil {
		return nil, err
	}

	return slot, nil
}
func (s *TimeSlotService) Delete(id string) error {
	return s.repo.Delete(id)
}

func (s *TimeSlotService) ListAll() ([]appointment.TimeSlot, error) {
	return s.repo.ListAll()
}

func (s *TimeSlotService) GetByDoctorAndDate(doctorID string, date time.Time) ([]appointment.TimeSlot, error) {
	return s.repo.FindByDoctorAndDate(doctorID, date)
}

func (s *TimeSlotService) GetByDoctorAndMonth(doctorID string, date time.Time) ([]appointment.TimeSlot, error) {
	return s.repo.FindByDoctorAndMonth(doctorID, date)
}

// ---------------- Helper ----------------
func generateSlotID() string {
	return uuid.NewString()
}
