package uploadservice

import (
	"fmt"
	"hospital-service/internal/storage"
	"mime/multipart"
	"path/filepath"
	"time"
)

type UploadService struct {
	storage *storage.S3Client
}

func NewUploadService(storage *storage.S3Client) *UploadService {
	return &UploadService{storage: storage}
}

func (u *UploadService) UploadFile(fileHeader *multipart.FileHeader) (string, error) {
	if fileHeader.Size > 5<<20 {
		return "", fmt.Errorf("file too large (max 5MB)")
	}

	key := fmt.Sprintf("uploads/%d_%s", time.Now().UnixNano(), filepath.Base(fileHeader.Filename))
	url, err := u.storage.UploadFile(fileHeader, key)
	if err != nil {
		return "", fmt.Errorf("failed to upload file: %v", err)
	}
	return url, nil
}

