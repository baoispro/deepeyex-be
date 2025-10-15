package medicalrecordrepo

import (
	"hospital-service/internal/models/medicalrecord"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type MedicalRecordRepo struct {
	db *gorm.DB
}

func NewMedicalRecordRepository(db *gorm.DB) *MedicalRecordRepo {
	return &MedicalRecordRepo{db: db}
}


// InitRecordAndDiagnosis: tạo MedicalRecord và AIDiagnosis trong cùng 1 transaction
func (r *MedicalRecordRepo) InitRecordAndDiagnosis(
	patientID, diseaseCode, diagnosisText string,
	confidence float64,
	mainImageURL string,
	eyeType, notes *string,
) (*medicalrecord.MedicalRecord, *medicalrecord.AIDiagnosis, error) {
	var (
		record *medicalrecord.MedicalRecord
		aiDiag *medicalrecord.AIDiagnosis
	)

	err := r.db.Transaction(func(tx *gorm.DB) error {
		// Step 1: Tạo MedicalRecord
		record = &medicalrecord.MedicalRecord{
			RecordID:  uuid.New().String(),
			PatientID: patientID,
			Diagnosis: diagnosisText,
		}
		if err := tx.Create(record).Error; err != nil {
			return err
		}

		// Step 2: Tạo AIDiagnosis với hình ảnh
		aiDiag = &medicalrecord.AIDiagnosis{
			ID:           uuid.New().String(),
			RecordID:     &record.RecordID,
			DiseaseCode:  diseaseCode,
			Confidence:   confidence,
			MainImageURL: mainImageURL, // ✅ Thêm
			EyeType:      eyeType,      // ✅ Thêm
			Notes:        notes,        // ✅ Thêm
		}
		if err := tx.Create(aiDiag).Error; err != nil {
			return err
		}

		return nil
	})

	if err != nil {
		return nil, nil, err
	}

	return record, aiDiag, nil
}

// ---------------- Create record ----------------
func (r *MedicalRecordRepo) Create(record *medicalrecord.MedicalRecord) error {
	return r.db.Create(record).Error
}

// ---------------- Get record by ID ----------------
func (r *MedicalRecordRepo) GetByID(id string) (*medicalrecord.MedicalRecord, error) {
	var record medicalrecord.MedicalRecord
	err := r.db.Preload("AI_Diagnoses.RecommendedPlans").
		Preload("Attachments").
		Preload("FollowUps").
		First(&record, "record_id = ?", id).Error
	return &record, err
}

// ---------------- List all records ----------------
func (r *MedicalRecordRepo) List() ([]*medicalrecord.MedicalRecord, error) {
	var records []*medicalrecord.MedicalRecord
	err := r.db.Preload("AI_Diagnoses.RecommendedPlans").
		Preload("Attachments").
		Preload("FollowUps").
		Find(&records).Error
	return records, err
}

// ---------------- Update record ----------------
func (r *MedicalRecordRepo) Update(record *medicalrecord.MedicalRecord) error {
	return r.db.Save(record).Error
}

// ---------------- Delete record ----------------
func (r *MedicalRecordRepo) Delete(id string) error {
	return r.db.Delete(&medicalrecord.MedicalRecord{}, "record_id = ?", id).Error
}


// ---------------------------- AI Diagnosis Management ----------------------------
// ---------------- Add AI Diagnosis ----------------
func (r *MedicalRecordRepo) AddAIDiagnosis(diag *medicalrecord.AIDiagnosis) (*medicalrecord.AIDiagnosis, error) {
	if err := r.db.Create(diag).Error; err != nil {
		return nil, err
	}
	return diag, nil
}
// ---------------- List AI Diagnoses by RecordID ----------------
func (r *MedicalRecordRepo) ListAIDiagnosesByRecordID(recordID string) ([]*medicalrecord.AIDiagnosis, error) {
	var diagnoses []*medicalrecord.AIDiagnosis
	err := r.db.Where("record_id = ?", recordID).Preload("RecommendedPlans").Find(&diagnoses).Error
	return diagnoses, err
}

// ---------------- GET AI Diagnosis by ID ----------------
func (r *MedicalRecordRepo) GetAIDiagnosisByID(id string) (*medicalrecord.AIDiagnosis, error) {
	var diag medicalrecord.AIDiagnosis
	err := r.db.Preload("RecommendedPlans").First(&diag, "id = ?", id).Error
	return &diag, err
}

// ---------------- Delete AI Diagnosis ----------------
func (r *MedicalRecordRepo) DeleteAIDiagnosis(id string) error {
	return r.db.Delete(&medicalrecord.AIDiagnosis{}, "id = ?", id).Error
}
