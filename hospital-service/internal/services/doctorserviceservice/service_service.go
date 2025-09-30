package doctorserviceservice

import (
	"hospital-service/internal/models/service"
	"hospital-service/internal/repositories/servicerepo"
)

type ServiceService struct {
	repo *servicerepo.ServiceRepo
}

func NewServiceService(repo *servicerepo.ServiceRepo) *ServiceService {
	return &ServiceService{repo: repo}
}

// ---------------- GetByDoctorID ----------------
// Lấy tất cả service theo doctor_id
func (s *ServiceService) GetServicesByDoctorID(doctorID string) ([]service.Service, error) {
	return s.repo.GetServicesByDoctorID(doctorID)
}
