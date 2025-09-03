package hospitalservice

import (
	"hospital-service/internal/models/hospital"
	"hospital-service/internal/repositories/hospitalrepo"

	"github.com/google/uuid"
)

// HospitalService xử lý logic nghiệp vụ
type HospitalService struct {
	hospitalRepo *hospitalrepo.HospitalRepo
}

func NewHospitalService(repo *hospitalrepo.HospitalRepo) *HospitalService {
	return &HospitalService{hospitalRepo: repo}
}

// CreateHospital
func (s *HospitalService) CreateHospital(name, address, phone, email string) (*hospital.Hospital, error) {
	h := &hospital.Hospital{
		HospitalID: generatePatientID(),
		Name:       name,
		Address:    address,
		Phone:      phone,
		Email:      email,
	}
	err := s.hospitalRepo.Create(h)
	return h, err
}

// GetHospitalByID
func (s *HospitalService) GetHospitalByID(id string) (*hospital.Hospital, error) {
	return s.hospitalRepo.FindByID(id)
}

// UpdateHospital
func (s *HospitalService) UpdateHospital(h *hospital.Hospital) error {
	return s.hospitalRepo.Update(h)
}

// DeleteHospital
func (s *HospitalService) DeleteHospital(id string) error {
	return s.hospitalRepo.Delete(id)
}

// ListHospitals
func (s *HospitalService) ListHospitals() ([]hospital.Hospital, error) {
	return s.hospitalRepo.List()
}

// ---------------- Helper ----------------
func generatePatientID() string {
	// Có thể dùng uuid.NewString() từ "github.com/google/uuid"
	// Hoặc logic riêng nếu muốn
	return uuid.NewString()
}
