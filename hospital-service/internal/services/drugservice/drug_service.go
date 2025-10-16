package drugservice

import (
	"errors"
	"hospital-service/internal/models/drug"
	"hospital-service/internal/repositories/drugrepo"
	"hospital-service/internal/storage"
	"path/filepath"
	"time"

	"github.com/google/uuid"
)

type DrugService struct {
	repo    *drugrepo.DrugRepo
	storage *storage.S3Client
}

func NewDrugService(repo *drugrepo.DrugRepo, storage *storage.S3Client) *DrugService {
	return &DrugService{repo: repo, storage: storage}
}

// ---------------- CreateDrug ----------------
func (s *DrugService) CreateDrug(name, description string, price float64, stock int, discount float64, imageFile interface{}) (*drug.Drug, error) {
	if name == "" || price <= 0 || stock < 0 {
		return nil, errors.New("invalid drug data")
	}

	var imageURL string
	if imageFile != nil {
		fileHeader := imageFile.(*storage.FileHeader)
		key := "drugs/" + uuid.NewString() + filepath.Ext(fileHeader.Filename)
		url, err := s.storage.UploadFile(fileHeader, key)
		if err != nil {
			return nil, err
		}
		imageURL = url
	}

	d := &drug.Drug{
		DrugID:          generateDrugID(),
		Name:            name,
		Description:     description,
		Price:           price,
		StockQuantity:   stock,
		DiscountPercent: discount,
		Image:           imageURL,
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}

	if err := s.repo.Create(d); err != nil {
		return nil, err
	}

	return d, nil
}

// ---------------- GetDrug ----------------
func (s *DrugService) GetDrug(id string) (*drug.Drug, error) {
	return s.repo.GetByID(id)
}

// ---------------- ListDrugs ----------------
func (s *DrugService) ListDrugs(name, minPrice, maxPrice, minStock, maxStock string) ([]drug.Drug, error) {
	return s.repo.FindWithFilters(name, minPrice, maxPrice, minStock, maxStock)
}

// ---------------- UpdateDrug ----------------
func (s *DrugService) UpdateDrug(d *drug.Drug, imageFile interface{}) error {
	if imageFile != nil {
		fileHeader := imageFile.(*storage.FileHeader)
		key := "drugs/" + uuid.NewString() + filepath.Ext(fileHeader.Filename)
		url, err := s.storage.UploadFile(fileHeader, key)
		if err != nil {
			return err
		}
		d.Image = url
	}
	d.UpdatedAt = time.Now()
	return s.repo.Update(d)
}

// ---------------- DeleteDrug ----------------
func (s *DrugService) DeleteDrug(id string) error {
	return s.repo.Delete(id)
}

// ---------------- Helper ----------------
func generateDrugID() string {
	return uuid.NewString()
}
