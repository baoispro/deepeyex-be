package medicalrecordservice

import (
	"errors"
	"hospital-service/internal/models/medicalrecord"
	"hospital-service/internal/repositories/medicalrecordrepo"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type MedicalRecordService struct {
	repo *medicalrecordrepo.MedicalRecordRepo
}

func NewMedicalRecordService(repo *medicalrecordrepo.MedicalRecordRepo) *MedicalRecordService {
	return &MedicalRecordService{repo: repo}
}

// ---------------- MedicalRecord Management ----------------
func (s *MedicalRecordService) InitRecordAndDiagnosis(req medicalrecord.InitRecordAndDiagnosisRequest) (*medicalrecord.InitRecordAndDiagnosisResponse, error) {
	record, aiDiag, err := s.repo.InitRecordAndDiagnosis(
		req.PatientID,
		req.DiseaseCode,
		req.Diagnosis,
		req.Confidence,
		req.MainImageURL, // ✅ Thêm hình ảnh
		req.EyeType,      // ✅ Thêm loại mắt
		req.Notes,        // ✅ Thêm ghi chú
	)
	if err != nil {
		return nil, err
	}

	return &medicalrecord.InitRecordAndDiagnosisResponse{
		RecordID:  record.RecordID,
		PatientID: record.PatientID,
		CreatedAt: record.CreatedAt,
		Diagnosis: *aiDiag,
	}, nil
}

func (s *MedicalRecordService) CreateRecord(
	patientID, doctorID, diagnosis, appointmentID string,
	notes *string,
	relatedRecordID *string,
) (*medicalrecord.MedicalRecord, error) {

	if patientID == "" || doctorID == "" || diagnosis == "" {
		return nil, errors.New("missing required fields: patient_id, doctor_id, diagnosis")
	}

	record := &medicalrecord.MedicalRecord{
		RecordID:        uuid.New().String(),
		PatientID:       patientID,
		AppointmentID:   appointmentID,
		DoctorID:        doctorID,
		Diagnosis:       diagnosis,
		Notes:           notes,
		RelatedRecordID: relatedRecordID,
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}

	if err := s.repo.Create(record); err != nil {
		return nil, err
	}

	return s.repo.GetByID(record.RecordID) // preload relations nếu cần
}

func (s *MedicalRecordService) GetRecord(id string) (*medicalrecord.MedicalRecord, error) {
	return s.repo.GetByID(id)
}

func (s *MedicalRecordService) ListRecords() ([]*medicalrecord.MedicalRecord, error) {
	return s.repo.List()
}

func (s *MedicalRecordService) UpdateRecord(record *medicalrecord.MedicalRecord) error {
	record.UpdatedAt = time.Now()
	return s.repo.Update(record)
}

func (s *MedicalRecordService) DeleteRecord(id string) error {
	return s.repo.Delete(id)
}

func (s *MedicalRecordService) CheckRecordByAppointment(appointmentID string) (*medicalrecord.MedicalRecord, bool, error) {
	// appointment_id bắt buộc
	if appointmentID == "" {
		return nil, false, errors.New("appointment_id is required")
	}

	record, err := s.repo.GetByAppointmentID(appointmentID)
	if err != nil && !errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, false, err
	}

	if record != nil {
		// Nếu đã có record → update
		return record, true, nil
	}

	// Chưa có record → tạo mới
	return nil, false, nil
}

// ---------------- Get all MedicalRecords by PatientID ----------------
func (s *MedicalRecordService) GetRecordsByPatient(patientID string) ([]*medicalrecord.MedicalRecord, error) {
	if patientID == "" {
		return nil, errors.New("patient_id is required")
	}

	records, err := s.repo.GetByPatientID(patientID)
	if err != nil {
		return nil, err
	}

	return records, nil
}
