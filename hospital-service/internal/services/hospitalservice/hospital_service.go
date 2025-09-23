package hospitalservice

import (
	"hospital-service/internal/models/hospital"
	"hospital-service/internal/repositories/hospitalrepo"
	"hospital-service/internal/storage"
	"path/filepath"

	"github.com/google/uuid"
	"github.com/gosimple/slug"
)

type HospitalService struct {
	hospitalRepo *hospitalrepo.HospitalRepo
	storage      *storage.S3Client
}

func NewHospitalService(repo *hospitalrepo.HospitalRepo, storage *storage.S3Client) *HospitalService {
	return &HospitalService{hospitalRepo: repo, storage: storage}
}

// ---------------- CreateHospital ----------------
func (s *HospitalService) CreateHospital(name, address, phone, email, urlMap, ward, city string, logoFile interface{}, latitude, longitude float64) (*hospital.Hospital, error) {
	var logoURL string
	if logoFile != nil {
		fileHeader := logoFile.(*storage.FileHeader)
		key := "hospitals/" + uuid.NewString() + filepath.Ext(fileHeader.Filename)

		url, err := s.storage.UploadFile(fileHeader, key)
		if err != nil {
			return nil, err
		}
		logoURL = url
	}

	h := &hospital.Hospital{
		HospitalID: generateHospitalID(),
		Name:       name,
		Address:    address,
		Phone:      phone,
		Email:      email,
		Image:      logoURL,
		Slug:       slug.Make(name),
		UrlMap:     urlMap,
		Ward:       ward,
		City:       city,
		Latitude:   latitude,
		Longitude:  longitude,
	}
	err := s.hospitalRepo.Create(h)
	return h, err
}

// ---------------- GetHospitalByID ----------------
func (s *HospitalService) GetHospitalByID(id string) (*hospital.Hospital, error) {
	return s.hospitalRepo.FindByID(id)
}

// ---------------- UpdateHospital ----------------
func (s *HospitalService) UpdateHospital(h *hospital.Hospital, logoFile interface{}) error {
	if logoFile != nil {
		fileHeader := logoFile.(*storage.FileHeader)
		key := "hospitals/" + uuid.NewString() + filepath.Ext(fileHeader.Filename)

		url, err := s.storage.UploadFile(fileHeader, key)
		if err != nil {
			return err
		}
		h.Image = url
	}

	h.Slug = slug.Make(h.Name)

	return s.hospitalRepo.Update(h)
}

// ---------------- DeleteHospital ----------------
func (s *HospitalService) DeleteHospital(id string) error {
	return s.hospitalRepo.Delete(id)
}

// ---------------- ListHospitals ----------------
func (s *HospitalService) ListHospitals() ([]hospital.Hospital, error) {
	return s.hospitalRepo.List()
}

// ---------------- ListCities ----------------
func (s *HospitalService) ListCities() ([]string, error) {
	return s.hospitalRepo.ListCities()
}

// ---------------- ListWardsByCity ----------------
func (s *HospitalService) ListWardsByCity(city string) ([]string, error) {
	return s.hospitalRepo.ListWardsByCity(city)
}

// ---------------- SearchByAddress ----------------
func (s *HospitalService) SearchByAddress(keyword string) ([]hospital.Hospital, error) {
	return s.hospitalRepo.SearchByAddress(keyword)
}

// ---------------- ListByCityAndWard ----------------
func (s *HospitalService) ListByCityAndWard(city, ward string) ([]hospital.Hospital, error) {
	return s.hospitalRepo.ListByCityAndWard(city, ward)
}

// ---------------- FindNearbyHospitals ----------------
func (s *HospitalService) FindNearbyHospitals(lat, lng, radiusKm float64) ([]hospital.Hospital, error) {
	return s.hospitalRepo.FindNearby(lat, lng, radiusKm)
}

// ---------------- GetHospitalBySlug ----------------
func (s *HospitalService) GetHospitalBySlug(slug string) (*hospital.Hospital, error) {
	return s.hospitalRepo.FindBySlug(slug)
}

// ---------------- Helper ----------------
func generateHospitalID() string {
	return uuid.NewString()
}
