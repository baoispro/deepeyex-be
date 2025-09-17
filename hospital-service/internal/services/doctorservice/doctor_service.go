package doctorservice

import (
	"hospital-service/internal/enums"
	"hospital-service/internal/models/doctor"
	"hospital-service/internal/repositories/doctorrepo"
	"hospital-service/internal/storage"
	"path/filepath"

	"github.com/google/uuid"
)

type DoctorService struct {
	doctorRepo *doctorrepo.DoctorRepo
	storage    *storage.S3Client
}

func NewDoctorService(repo *doctorrepo.DoctorRepo, storage *storage.S3Client) *DoctorService {
	return &DoctorService{doctorRepo: repo, storage: storage}
}

// ---------------- CreateDoctor ----------------
func (s *DoctorService) CreateDoctor(userID, fullName string, specialty enums.Specialty,
	hospitalID, phone, email string, avatarFile interface{}) (*doctor.Doctor, error) {

	var avatarURL string
	if avatarFile != nil {
		fileHeader := avatarFile.(*storage.FileHeader)
		key := "doctors/" + uuid.NewString() + filepath.Ext(fileHeader.Filename)
		url, err := s.storage.UploadFile(fileHeader, key)
		if err != nil {
			return nil, err
		}
		avatarURL = url
	}

	d := &doctor.Doctor{
		DoctorID:   generateDoctorID(),
		UserID:     userID,
		FullName:   fullName,
		Specialty:  specialty,
		HospitalID: hospitalID,
		Phone:      phone,
		Email:      email,
		Image:      avatarURL,
	}
	err := s.doctorRepo.Create(d)
	return d, err
}

// ---------------- GetDoctorByID ----------------
func (s *DoctorService) GetDoctorByID(id string) (*doctor.Doctor, error) {
	return s.doctorRepo.FindByID(id)
}

// ---------------- UpdateDoctor ----------------
func (s *DoctorService) UpdateDoctor(d *doctor.Doctor, avatarFile interface{}) error {
	if avatarFile != nil {
		fileHeader := avatarFile.(*storage.FileHeader)
		key := "doctors/" + uuid.NewString() + filepath.Ext(fileHeader.Filename)

		url, err := s.storage.UploadFile(fileHeader, key)
		if err != nil {
			return err
		}
		d.Image = url
	}

	return s.doctorRepo.Update(d)
}

// ---------------- DeleteDoctor ----------------
func (s *DoctorService) DeleteDoctor(id string) error {
	return s.doctorRepo.Delete(id)
}

// ---------------- ListDoctors ----------------
func (s *DoctorService) ListDoctors() ([]doctor.Doctor, error) {
	return s.doctorRepo.List()
}

// ---------------- FindByHospitalID ----------------
func (s *DoctorService) FindByHospitalID(hospitalID string) ([]doctor.Doctor, error) {
	return s.doctorRepo.FindByHospitalID(hospitalID)
}

// ---------------- FindByUserID ----------------
func (s *DoctorService) FindByUserID(userID string) (*doctor.Doctor, error) {
	return s.doctorRepo.FindByUserID(userID)
}

// ---------------- Helper ----------------
func generateDoctorID() string {
	return uuid.NewString()
}
