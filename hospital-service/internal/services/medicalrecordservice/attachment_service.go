package medicalrecordservice

import (
	"errors"
	"hospital-service/internal/models/medicalrecord"
	"hospital-service/internal/repositories/medicalrecordrepo"
)

type AttachmentService struct {
	attachmentRepo *medicalrecordrepo.AttachmentRepo
}

// NewAttachmentService khởi tạo service
func NewAttachmentService(attachmentRepo *medicalrecordrepo.AttachmentRepo) *AttachmentService {
	return &AttachmentService{
		attachmentRepo: attachmentRepo,
	}
}

// ---------------------- AddAttachment ----------------------
func (s *AttachmentService) AddAttachment(att *medicalrecord.Attachment) error {
	if att.RecordID == "" {
		return errors.New("record_id is required")
	}
	if att.FileURL == "" {
		return errors.New("file_url is required")
	}

	return s.attachmentRepo.AddAttachment(att)
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
