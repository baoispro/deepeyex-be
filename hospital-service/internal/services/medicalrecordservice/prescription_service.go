package medicalrecordservice

import (
	"errors"
	"hospital-service/internal/models/medicalrecord"
	"hospital-service/internal/repositories/medicalrecordrepo"
	"time"

	"github.com/google/uuid"
)


type PrescriptionService struct {
	repo *medicalrecordrepo.PrescriptionRepo
}

func NewPrescriptionService(repo *medicalrecordrepo.PrescriptionRepo) *PrescriptionService {
	return &PrescriptionService{repo: repo}
}


func (s *PrescriptionService) CreatePrescription(status, recordId, approvedBy string, approvedAt *time.Time ) (*medicalrecord.Prescription,error) {
	
	if status == "" || recordId == "" {
		return nil, errors.New("missing required fields: status, record_id")
	}

	p := &medicalrecord.Prescription{
		PrescriptionID: uuid.New().String(),
		Status:         status,
		CreatedAt:      time.Now(),
		UpdatedAt:      time.Now(),
	}

	if err := s.repo.Create(p); err != nil {
		return nil, err
	}
	return p, nil
}

func (s *PrescriptionService) GetPrescriptionByID(id string) (*medicalrecord.Prescription, error) {
	return s.repo.GetPrescriptionByID(id)
}

func (s *PrescriptionService) GetPrescriptionsByMedicalRecordID(medicalRecordID string) ([]*medicalrecord.Prescription, error) {
	return s.repo.GetPrescriptionsByMedicalRecordID(medicalRecordID)
}

func (s *PrescriptionService) Approve(id, doctorID string) error {
	return s.repo.Approve(id, doctorID)
}

func (s *PrescriptionService) UpdatePrecription(p *medicalrecord.Prescription) error {
	return s.repo.UpdatePrecription(p)
}

func (s *PrescriptionService) Delete(id string) error {
	return s.repo.Delete(id)
}