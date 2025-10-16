package medicalrecordservice

import (
	"errors"
	"hospital-service/internal/models/medicalrecord"
	"hospital-service/internal/repositories/medicalrecordrepo"
	"hospital-service/internal/storage"
	"time"

	"github.com/google/uuid"
)

type AttachmentService struct {
	attachmentRepo *medicalrecordrepo.AttachmentRepo
	storage        *storage.S3Client
}

// NewAttachmentService khởi tạo service
func NewAttachmentService(attachmentRepo *medicalrecordrepo.AttachmentRepo, storage *storage.S3Client) *AttachmentService {
	return &AttachmentService{
		attachmentRepo: attachmentRepo,
		storage:        storage,
	}
}

// ---------------------- AddAttachment ----------------------
func (s *AttachmentService) AddAttachment(att *medicalrecord.Attachment, file interface{}) (*medicalrecord.Attachment, error) {
	if att.RecordID == "" {
		return nil, errors.New("record_id is required")
	}
	if att.FileType == "" {
		return nil, errors.New("file_type is required")
	}
	if file == nil {
		return nil, errors.New("file is required")
	}

	// upload file lên S3
	fileHeader := file.(*storage.FileHeader)
	key := "attachments/" + uuid.NewString() + "_" + fileHeader.Filename
	url, err := s.storage.UploadFile(fileHeader, key)
	if err != nil {
		return nil, err
	}

	att.FileURL = url
	att.ID = uuid.NewString()
	att.CreatedAt = time.Now()

	if err := s.attachmentRepo.AddAttachment(att); err != nil {
		return nil, err
	}

	return att, nil
}

// ---------------------- GetAttachmentsByRecordID ----------------------
func (s *AttachmentService) GetAttachmentsByRecordID(recordID string) ([]medicalrecord.Attachment, error) {
	if recordID == "" {
		return nil, errors.New("record_id is required")
	}
	return s.attachmentRepo.GetByRecordID(recordID)
}

// ---------------------- DeleteAttachmentByID ----------------------
func (s *AttachmentService) DeleteAttachmentByID(id string) error {
	if id == "" {
		return errors.New("id is required")
	}
	return s.attachmentRepo.DeleteByID(id)
}
