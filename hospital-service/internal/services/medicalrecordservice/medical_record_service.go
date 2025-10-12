package medicalrecordservice

import (
	"errors"
	"hospital-service/internal/models/medicalrecord"
	"hospital-service/internal/repositories/medicalrecordrepo"
	"time"

	"github.com/google/uuid"
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
	patientID, doctorID, diagnosis, createdBy, appointmentID string,
) (*medicalrecord.MedicalRecord, error) {

	if patientID == "" || doctorID == "" || diagnosis == "" || createdBy == "" {
		return nil, errors.New("missing required fields: patient_id, doctor_id, diagnosis, created_by")
	}

	record := &medicalrecord.MedicalRecord{
		RecordID:      uuid.New().String(),
		PatientID:     patientID,
		AppointmentID: appointmentID, 
		DoctorID:      doctorID,
		Diagnosis:     diagnosis,
		CreatedBy:     createdBy, 
		CreatedAt:     time.Now(),
		UpdatedAt:     time.Now(),
	}

	if err := s.repo.Create(record); err != nil {
		return nil, err
	}

	return record, nil
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


// ---------------- AIDiagnosis Management ----------------
func (s *MedicalRecordService) ListAIDiagnoses(recordID string) ([]*medicalrecord.AIDiagnosis, error) {
	return s.repo.ListAIDiagnosesByRecordID(recordID)
}

func (s *MedicalRecordService) AddAIDiagnosisByRecordID(
	recordID, diseaseCode string,
	confidence float64,
	mainImageURL string,
	eyeType, notes *string,
) (*medicalrecord.AIDiagnosis, error) {
	diagnosis := &medicalrecord.AIDiagnosis{
		ID:           uuid.New().String(),
		RecordID:     recordID,
		DiseaseCode:  diseaseCode,
		Confidence:   confidence,
		MainImageURL: mainImageURL, // ✅ Thêm
		EyeType:      eyeType,      // ✅ Thêm
		Notes:        notes,        // ✅ Thêm
		CreatedAt:    time.Now(),
	}

	return s.repo.AddAIDiagnosis(diagnosis)
}

func (s *MedicalRecordService) GetAIDiagnosisByID(id string) (*medicalrecord.AIDiagnosis, error) {
	return s.repo.GetAIDiagnosisByID(id)
}

func (s *MedicalRecordService) DeleteAIDiagnosis(id string) error {
	return s.repo.DeleteAIDiagnosis(id)
}

// ---------------- AI RecommendedPlan Management ----------------
func (s *MedicalRecordService) ListRecommendedPlans(diagnosisID string) ([]*medicalrecord.AIRecommendedPlan, error) {
	return s.repo.ListAIRecommendedPlansByDiagnosisID(diagnosisID)
}

func (s *MedicalRecordService) AddRecommendedPlan(diagnosisID, description, drugName, dosage, frequency string, durationDays int) (*medicalrecord.AIRecommendedPlan, error) {
	plan := &medicalrecord.AIRecommendedPlan{
		ID:          uuid.New().String(),
		AIDiagnosisID : diagnosisID,
		DurationDays: durationDays,
		DrugName:   drugName,
		Dosage:    dosage,
		Frequency: frequency,

	}
	return s.repo.AddAIRecommendedPlan(plan)
}

func (s *MedicalRecordService) DeleteRecommendedPlan(id string) error {
	return s.repo.DeleteAIRecommendedPlan(id)
}
