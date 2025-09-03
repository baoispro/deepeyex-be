package doctorservice

import (
	"hospital-service/internal/enums"
	"hospital-service/internal/models/doctor"
	"hospital-service/internal/repositories/doctorrepo"

	"github.com/google/uuid"
)

type DoctorService struct {
	doctorRepo *doctorrepo.DoctorRepo
}

func NewDoctorService(repo *doctorrepo.DoctorRepo) *DoctorService {
	return &DoctorService{doctorRepo: repo}
}

// ---------------- CreateDoctor ----------------
func (s *DoctorService) CreateDoctor(userID, fullName string, specialty enums.Specialty, hospitalID, phone, email string) (*doctor.Doctor, error) {
	d := &doctor.Doctor{
		DoctorID:  generateDoctorID(), // tạo ID mới
		UserID:    userID,
		FullName:  fullName,
		Specialty: specialty,

		HospitalID: hospitalID, // thay thế hospital bằng hospital_id
		Phone:     phone,
		Email:     email,
	}
	err := s.doctorRepo.Create(d)
	return d, err
}

// ---------------- GetDoctorByID ----------------
func (s *DoctorService) GetDoctorByID(id string) (*doctor.Doctor, error) {
	return s.doctorRepo.FindByID(id)
}

// ---------------- UpdateDoctor ----------------
func (s *DoctorService) UpdateDoctor(d *doctor.Doctor) error {
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

//----------------- FindByUserID ----------------
func (s *DoctorService) FindByUserID(userID string) (*doctor.Doctor, error) {
	return s.doctorRepo.FindByUserID(userID)
}

// ---------------- Helper ----------------
func generateDoctorID() string {
	return uuid.NewString()
}
