package medicalrecordservice

import (
	"errors"
	"hospital-service/internal/models/medicalrecord"
	"hospital-service/internal/repositories/medicalrecordrepo"
	"time"

	"github.com/google/uuid"
)


type FollowUpService struct {
	repo *medicalrecordrepo.FollowUpRepo
}

func NewFollowUpService(r *medicalrecordrepo.FollowUpRepo) *FollowUpService {
	return &FollowUpService{repo: r}
}

// -------------------- Create FollowUp --------------------
func (s *FollowUpService) CreateFollowUp(recordID, note, createdBy string, nextAppointment *time.Time) (*medicalrecord.FollowUp, error) {
	if recordID == "" {
		return nil, errors.New("record_id is required")
	}

	followUp := &medicalrecord.FollowUp{
		FollowUpID:   uuid.New().String(),
		RecordID:     recordID,
		NextAppointment: nextAppointment,
		CreatedAt:    time.Now(),
		Notes: 	 note,
	
	}

	if err := s.repo.AddFollowUp(followUp); err != nil {
		return nil, err
	}

	return followUp, nil
}

// -------------------- Get FollowUps by RecordID --------------------
func (s *FollowUpService) GetFollowUpsByRecordID(recordID string) ([]medicalrecord.FollowUp, error) {
	if recordID == "" {
		return nil, errors.New("record_id is required")
	}
	return s.repo.GetByRecordID(recordID)
}

// -------------------- Update FollowUp --------------------
func (s *FollowUpService) UpdateFollowUp(followUpID, note string, nextAppointment *time.Time) (*medicalrecord.FollowUp, error) {
	if followUpID == "" {
		return nil, errors.New("follow_up_id is required")
	}

	existing, err := s.getFollowUpByID(followUpID)
	if err != nil {
		return nil, err
	}

	// Cập nhật dữ liệu
	existing.Notes = note
	existing.NextAppointment = nextAppointment


	if err := s.repo.Update(existing); err != nil {
		return nil, err
	}

	return existing, nil
}

// -------------------- Delete FollowUp --------------------
func (s *FollowUpService) DeleteFollowUp(followUpID string) error {
	if followUpID == "" {
		return errors.New("follow_up_id is required")
	}
	return s.repo.DeleteByID(followUpID)
}

// -------------------- Helper: get FollowUp by ID --------------------
func (s *FollowUpService) getFollowUpByID(followUpID string) (*medicalrecord.FollowUp, error) {
	if followUpID == "" {
		return nil, errors.New("follow_up_id is required")
	}

	// Sử dụng GetByRecordID để tìm kiếm (nếu bạn chưa có hàm GetByID riêng trong repo)
	// Ở đây tốt nhất là viết hàm repo.GetByID
	var followUps []medicalrecord.FollowUp
	followUps, err := s.repo.GetByRecordID(followUpID)
	if err != nil {
		return nil, err
	}
	if len(followUps) == 0 {
		return nil, errors.New("follow_up not found")
	}

	return &followUps[0], nil
}