package medicalrecordservice

import (
	"errors"
	"fmt"
	"hospital-service/internal/models/medicalrecord"
	"hospital-service/internal/repositories/medicalrecordrepo"
	"hospital-service/internal/storage"
	"path/filepath"
	"time"

	"github.com/google/uuid"
)

type AIDiagnosisService struct {
	repo    *medicalrecordrepo.AIDiagnosisRepo
	storage *storage.S3Client
}

// NewAIDiagnosisService khởi tạo service
func NewAIDiagnosisService(repo *medicalrecordrepo.AIDiagnosisRepo, storage *storage.S3Client) *AIDiagnosisService {
	return &AIDiagnosisService{
		repo:    repo,
		storage: storage,
	}
}

// ✅ Request khi AI tạo bản chẩn đoán
type CreateAIDiagnosisRequest struct {
	PatientID    string  `json:"patient_id" binding:"required"`
	RecordID     string  `json:"record_id" binding:"required"`
	DiseaseCode  string  `json:"disease_code" binding:"required"`
	Confidence   float64 `json:"confidence" binding:"required"`
	MainImageURL string  `json:"main_image_url"`
	EyeType      *string `json:"eye_type,omitempty"`
	Notes        *string `json:"notes,omitempty"`
}

// ✅ Response trả về
type AIDiagnosisResponse struct {
	ID           string     `json:"id"`
	PatientID    string     `json:"patient_id"`
	RecordID     string     `json:"record_id"`
	DiseaseCode  string     `json:"disease_code"`
	Confidence   float64    `json:"confidence"`
	MainImageURL string     `json:"main_image_url"`
	EyeType      *string    `json:"eye_type,omitempty"`
	Notes        *string    `json:"notes,omitempty"`
	Status       string     `json:"status"`
	CreatedAt    time.Time  `json:"created_at"`
	VerifiedBy   *string    `json:"verified_by,omitempty"`
	VerifiedAt   *time.Time `json:"verified_at,omitempty"`
}

// Create tạo mới chẩn đoán AI
func (s *AIDiagnosisService) Create(req CreateAIDiagnosisRequest, mainImageFile interface{}) (*AIDiagnosisResponse, error) {
	if req.PatientID == "" || req.DiseaseCode == "" {
		return nil, errors.New("missing required fields")
	}

	var mainImageURL string
	if mainImageFile != nil {
		fileHeader := mainImageFile.(*storage.FileHeader)
		key := "ai_diagnosis/" + uuid.NewString() + filepath.Ext(fileHeader.Filename)
		url, err := s.storage.UploadFile(fileHeader, key)
		if err != nil {
			return nil, fmt.Errorf("failed to upload image: %v", err)
		}
		mainImageURL = url
	}

	var recordID *string
	if req.RecordID != "" {
		recordID = &req.RecordID
	}

	diagnosis := &medicalrecord.AIDiagnosis{
		ID:           uuid.NewString(),
		PatientID:    req.PatientID,
		RecordID:     recordID,
		DiseaseCode:  req.DiseaseCode,
		Confidence:   req.Confidence,
		MainImageURL: mainImageURL,
		EyeType:      req.EyeType,
		Notes:        req.Notes,
		Status:       "PENDING",
	}

	if err := s.repo.Create(diagnosis); err != nil {
		return nil, fmt.Errorf("failed to create ai diagnosis: %v", err)
	}

	return &AIDiagnosisResponse{
		ID:        diagnosis.ID,
		PatientID: diagnosis.PatientID,
		RecordID: func() string {
			if diagnosis.RecordID == nil {
				return ""
			}
			return *diagnosis.RecordID
		}(),
		DiseaseCode:  diagnosis.DiseaseCode,
		Confidence:   diagnosis.Confidence,
		MainImageURL: diagnosis.MainImageURL,
		EyeType:      diagnosis.EyeType,
		Notes:        diagnosis.Notes,
		Status:       diagnosis.Status,
		CreatedAt:    diagnosis.CreatedAt,
	}, nil
}

// FindByPatientID lấy danh sách chẩn đoán AI theo bệnh nhân
func (s *AIDiagnosisService) FindByPatientID(patientID string) ([]medicalrecord.AIDiagnosis, error) {
	if patientID == "" {
		return nil, errors.New("patientID is required")
	}
	return s.repo.FindByPatientID(patientID)
}

// FindAll lấy toàn bộ chẩn đoán AI có status Pending
func (s *AIDiagnosisService) FindAll() ([]medicalrecord.AIDiagnosis, error) {
	return s.repo.FindAllPending()
}

// FindAllApproved lấy toàn bộ chẩn đoán AI có status APPROVED
func (s *AIDiagnosisService) FindAllApproved() ([]medicalrecord.AIDiagnosis, error) {
	return s.repo.FindAllApproved()
}

// Verify cho bác sĩ xác nhận chẩn đoán AI
func (s *AIDiagnosisService) Verify(id, doctorID, status, notes string, signatureFile interface{}) error {
	diagnosis, err := s.repo.FindByID(id)
	if err != nil {
		return fmt.Errorf("diagnosis not found: %v", err)
	}

	now := time.Now()
	diagnosis.VerifiedBy = &doctorID
	diagnosis.VerifiedAt = &now
	diagnosis.Status = status
	if notes != "" {
		diagnosis.DoctorNotes = &notes
	}

	// Upload signature lên S3 nếu có
	if signatureFile != nil {
		fileHeader := signatureFile.(*storage.FileHeader)
		key := "signatures/" + uuid.NewString() + filepath.Ext(fileHeader.Filename)
		signatureURL, err := s.storage.UploadFile(fileHeader, key)
		if err != nil {
			return fmt.Errorf("failed to upload signature: %v", err)
		}
		diagnosis.VerificationSig = &signatureURL
	}

	if err := s.repo.Update(diagnosis); err != nil {
		return fmt.Errorf("failed to update diagnosis: %v", err)
	}

	return nil
}

// ---------------- Helper ----------------
func generateID() string {
	return uuid.NewString()
}
