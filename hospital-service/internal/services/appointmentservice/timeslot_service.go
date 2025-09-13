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

func (s *TimeSlotService) Update(slot *appointment.TimeSlot) error {
	return s.repo.Update(slot)
}

func (s *TimeSlotService) Delete(id string) error {
	return s.repo.Delete(id)
}

func (s *TimeSlotService) ListAll() ([]appointment.TimeSlot, error) {
	return s.repo.ListAll()
}


// ---------------- Helper ----------------
func generateSlotID() string {
	return uuid.NewString()
}
