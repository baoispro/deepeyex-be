package hospitalservice

import (
	"hospital-service/internal/models/hospital"
	"hospital-service/internal/repositories/hospitalrepo"
	"hospital-service/internal/storage"
	"path/filepath"

	"github.com/google/uuid"
)

type HospitalService struct {
	hospitalRepo *hospitalrepo.HospitalRepo
	storage      *storage.S3Client
}

func NewHospitalService(repo *hospitalrepo.HospitalRepo, storage *storage.S3Client) *HospitalService {
	return &HospitalService{hospitalRepo: repo, storage: storage}
}

// ---------------- CreateHospital ----------------
func (s *HospitalService) CreateHospital(name, address, phone, email string, logoFile interface{}) (*hospital.Hospital, error) {
	var logoURL string
	if logoFile != nil {
		fileHeader := logoFile.(*storage.FileHeader)
		key := "hospitals/" + uuid.NewString() + filepath.Ext(fileHeader.Filename)

		url, err := s.storage.UploadFile(fileHeader, key)
		if err != nil {
			return nil, err
		}
		logoURL = url
	}

	h := &hospital.Hospital{
		HospitalID: generateHospitalID(),
		Name:       name,
		Address:    address,
		Phone:      phone,
		Email:      email,
		Image:      logoURL,
	}
	err := s.hospitalRepo.Create(h)
	return h, err
}

// ---------------- GetHospitalByID ----------------
func (s *HospitalService) GetHospitalByID(id string) (*hospital.Hospital, error) {
	return s.hospitalRepo.FindByID(id)
}

// ---------------- UpdateHospital ----------------
func (s *HospitalService) UpdateHospital(h *hospital.Hospital, logoFile interface{}) error {
	if logoFile != nil {
		fileHeader := logoFile.(*storage.FileHeader)
		key := "hospitals/" + uuid.NewString() + filepath.Ext(fileHeader.Filename)

		url, err := s.storage.UploadFile(fileHeader, key)
		if err != nil {
			return err
		}
		h.Image = url
	}

	return s.hospitalRepo.Update(h)
}

// ---------------- DeleteHospital ----------------
func (s *HospitalService) DeleteHospital(id string) error {
	return s.hospitalRepo.Delete(id)
}

// ---------------- ListHospitals ----------------
func (s *HospitalService) ListHospitals() ([]hospital.Hospital, error) {
	return s.hospitalRepo.List()
}

// ---------------- Helper ----------------
func generateHospitalID() string {
	return uuid.NewString()
}
