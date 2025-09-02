package patientservice

import (
	"hospital-service/internal/enums"
	"hospital-service/internal/models/patient"
	"hospital-service/internal/repositories/patientrepo"
	"time"

	"github.com/google/uuid"
)

type PatientService struct {
	patientRepo *patientrepo.PatientRepo
}

func NewPatientService(repo *patientrepo.PatientRepo) *PatientService {
	return &PatientService{patientRepo: repo}
}

// ---------------- CreatePatient ----------------
func (s *PatientService) CreatePatient(userID, fullName string, dob time.Time, gender enums.Gender, address, phone, email string) (*patient.Patient, error) {
	p := &patient.Patient{
		PatientID: generatePatientID(), // bạn có thể dùng uuid.NewString() nếu muốn
		UserID:    userID,
		FullName:  fullName,
		DOB:       dob,
		Gender:    gender,
		Address:   address,
		Phone:     phone,
		Email:     email,
	}
	err := s.patientRepo.Create(p)
	return p, err
}

// ---------------- GetPatientByUserID ----------------
func (s *PatientService) GetPatientByUserID(userID string) (*patient.Patient, error) {
	return s.patientRepo.FindByUserID(userID)
}

// ---------------- GetPatientByID ----------------
func (s *PatientService) GetPatientByID(id string) (*patient.Patient, error) {
	return s.patientRepo.FindByID(id)
}

// ---------------- UpdatePatient ----------------
func (s *PatientService) UpdatePatient(p *patient.Patient) error {
	return s.patientRepo.Update(p)
}

// ---------------- DeletePatient ----------------
func (s *PatientService) DeletePatient(id string) error {
	return s.patientRepo.Delete(id)
}

// ---------------- ListPatients ----------------
func (s *PatientService) ListPatients() ([]patient.Patient, error) {
	return s.patientRepo.List()
}

// ---------------- Helper ----------------
func generatePatientID() string {
	// Có thể dùng uuid.NewString() từ "github.com/google/uuid"
	// Hoặc logic riêng nếu muốn
	return uuid.NewString()
}
