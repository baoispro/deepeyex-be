package hospitalhandler

import (
	"hospital-service/internal/config"
	"hospital-service/internal/services/hospitalservice"
	"hospital-service/internal/utils"
	"mime/multipart"
	"net/http"

	"github.com/gin-gonic/gin"
)

// HospitalHandler quản lý API endpoint cho Hospital
type HospitalHandler struct {
	service *hospitalservice.HospitalService
	cfg     config.Config
}

// NewHospitalHandler khởi tạo handler mới
func NewHospitalHandler(cfg config.Config, service *hospitalservice.HospitalService) *HospitalHandler {
	return &HospitalHandler{service: service, cfg: cfg}
}

// ----------- Request Structs -----------

type createHospitalReq struct {
	Name      string                `form:"name" binding:"required"`
	Address   string                `form:"address"`
	Phone     string                `form:"phone"`
	Email     string                `form:"email"`
	Logo      *multipart.FileHeader `form:"logo"`
	UrlMap    string                `form:"url_map"`
	Ward      string                `form:"ward"`
	City      string                `form:"city"`
	Latitude  float64               `form:"latitude" binding:"required"`
	Longitude float64               `form:"longitude" binding:"required"`
}

type updateHospitalReq struct {
	Name      string                `form:"name"`
	Address   string                `form:"address"`
	Phone     string                `form:"phone"`
	Email     string                `form:"email"`
	Logo      *multipart.FileHeader `form:"logo"`
	UrlMap    string                `form:"url_map"`
	Ward      string                `form:"ward"`
	City      string                `form:"city"`
	Latitude  *float64              `form:"latitude"`
	Longitude *float64              `form:"longitude"`
}

// ----------- Request Structs -----------
type nearbyHospitalReq struct {
	Latitude  float64 `json:"latitude" binding:"required"`
	Longitude float64 `json:"longitude" binding:"required"`
	RadiusKm  float64 `json:"radius_km" binding:"required"`
}

// ---------------- Create Hospital ----------------
// @Summary Create a new hospital
// @Description Add hospital info with optional logo upload
// @Tags Hospitals
// @Accept multipart/form-data
// @Produce json
// @Param name formData string true "Hospital Name"
// @Param address formData string false "Address"
// @Param phone formData string false "Phone"
// @Param email formData string false "Email"
// @Param url_map formData string false "Url Map"
// @Param ward formData string false "Ward"
// @Param city formData string false "City"
// @Param logo formData file false "Hospital Logo"
// @Param latitude formData number true "Latitude"
// @Param longitude formData number true "Longitude"
// @Success 201 {object} utils.APIResponse
// @Failure 400 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /hospitals [post]
func (h *HospitalHandler) CreateHospital(c *gin.Context) {
	var req createHospitalReq
	if err := c.ShouldBind(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	var logoFile interface{}
	if req.Logo != nil {
		logoFile = req.Logo
	}

	hospital, err := h.service.CreateHospital(req.Name, req.Address, req.Phone, req.Email, req.UrlMap, req.Ward, req.City, logoFile, req.Latitude, req.Longitude)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Hospital created successfully", hospital))
}

// ---------------- Get Hospital By ID ----------------
// @Summary Get hospital by ID
// @Description Retrieve hospital record using hospital ID
// @Tags Hospitals
// @Produce json
// @Param hospital_id path string true "Hospital ID"
// @Success 200 {object} hospital.Hospital
// @Failure 404 {object} map[string]string
// @Router /hospitals/{hospital_id} [get]
func (h *HospitalHandler) GetHospitalByID(c *gin.Context) {
	hospitalID := c.Param("hospital_id")
	hospitalData, err := h.service.GetHospitalByID(hospitalID)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Hospital not found"))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Hospital retrieved successfully", hospitalData))
}

// ---------------- Update Hospital ----------------
// @Summary Update a hospital
// @Description Update hospital info and logo
// @Tags Hospitals
// @Accept multipart/form-data
// @Produce json
// @Param hospital_id path string true "Hospital ID"
// @Param name formData string false "Hospital Name"
// @Param address formData string false "Address"
// @Param phone formData string false "Phone"
// @Param email formData string false "Email"
// @Param url_map formData string false "Url Map"
// @Param ward formData string false "Ward"
// @Param city formData string false "City"
// @Param logo formData file false "Hospital Logo"
// @Param latitude formData number false "Latitude"
// @Param longitude formData number false "Longitude"
// @Success 200 {object} utils.APIResponse
// @Failure 400 {object} utils.APIResponse
// @Failure 404 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /hospitals/{hospital_id} [put]
func (h *HospitalHandler) UpdateHospital(c *gin.Context) {
	hospitalID := c.Param("hospital_id")
	var req updateHospitalReq

	if err := c.ShouldBind(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	hospitalData, err := h.service.GetHospitalByID(hospitalID)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Hospital not found"))
		return
	}

	if req.Name != "" {
		hospitalData.Name = req.Name
	}
	if req.Address != "" {
		hospitalData.Address = req.Address
	}
	if req.Phone != "" {
		hospitalData.Phone = req.Phone
	}
	if req.Email != "" {
		hospitalData.Email = req.Email
	}
	if req.UrlMap != "" {
		hospitalData.UrlMap = req.UrlMap
	}

	if req.Ward != "" {
		hospitalData.Ward = req.Ward
	}
	if req.City != "" {
		hospitalData.City = req.City
	}

	if req.Latitude != nil {
		hospitalData.Latitude = *req.Latitude
	}
	if req.Longitude != nil {
		hospitalData.Longitude = *req.Longitude
	}

	var logoFile interface{}
	if req.Logo != nil {
		logoFile = req.Logo
	}

	if err := h.service.UpdateHospital(hospitalData, logoFile); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Hospital updated successfully", hospitalData))
}

// ---------------- Delete Hospital ----------------
// @Summary Delete hospital
// @Description Delete hospital by hospital ID
// @Tags Hospitals
// @Produce json
// @Param hospital_id path string true "Hospital ID"
// @Success 200 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /hospitals/{hospital_id} [delete]
func (h *HospitalHandler) DeleteHospital(c *gin.Context) {
	hospitalID := c.Param("hospital_id")

	if err := h.service.DeleteHospital(hospitalID); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Hospital deleted successfully", nil))
}

// ---------------- List Hospitals ----------------
// @Summary List all hospitals
// @Description Retrieve all hospitals
// @Tags Hospitals
// @Produce json
// @Success 200 {array} hospital.Hospital
// @Failure 500 {object} map[string]string
// @Router /hospitals [get]
func (h *HospitalHandler) ListHospitals(c *gin.Context) {
	hospitals, err := h.service.ListHospitals()
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Hospitals retrieved successfully", hospitals))
}

// ---------------- List Cities ----------------
// @Summary List all cities
// @Description Retrieve all distinct cities that have hospitals
// @Tags Hospitals
// @Produce json
// @Success 200 {array} string
// @Failure 500 {object} utils.APIResponse
// @Router /hospitals/cities [get]
func (h *HospitalHandler) ListCities(c *gin.Context) {
	cities, err := h.service.ListCities()
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Cities retrieved successfully", cities))
}

// ---------------- List Wards By City ----------------
// @Summary List wards by city
// @Description Retrieve all distinct wards for a given city
// @Tags Hospitals
// @Produce json
// @Param city query string true "City name"
// @Success 200 {array} string
// @Failure 400 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /hospitals/wards [get]
func (h *HospitalHandler) ListWardsByCity(c *gin.Context) {
	city := c.Query("city")
	if city == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "City parameter is required"))
		return
	}

	wards, err := h.service.ListWardsByCity(city)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Wards retrieved successfully", wards))
}

// ---------------- Search By Address ----------------
// @Summary Search hospitals by address
// @Description Search hospitals by keyword in address, ward, or city
// @Tags Hospitals
// @Produce json
// @Param keyword query string true "Search keyword"
// @Success 200 {array} hospital.Hospital
// @Failure 400 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /hospitals/search [get]
func (h *HospitalHandler) SearchByAddress(c *gin.Context) {
	keyword := c.Query("keyword")
	if keyword == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Keyword parameter is required"))
		return
	}

	results, err := h.service.SearchByAddress(keyword)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Hospitals retrieved successfully", results))
}

// ---------------- List By City And Ward ----------------
// @Summary List hospitals by city and ward
// @Description Retrieve hospitals filtered by city and ward (both optional, can filter by either or both)
// @Tags Hospitals
// @Produce json
// @Param city query string false "City name"
// @Param ward query string false "Ward name"
// @Success 200 {array} hospital.Hospital
// @Failure 500 {object} utils.APIResponse
// @Router /hospitals/filter [get]
func (h *HospitalHandler) ListByCityAndWard(c *gin.Context) {
	city := c.Query("city")
	ward := c.Query("ward")

	results, err := h.service.ListByCityAndWard(city, ward)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Hospitals retrieved successfully", results))
}

// ---------------- Find Nearby Hospitals ----------------
// @Summary Find nearby hospitals
// @Description Find hospitals within a radius (km) from given coordinates
// @Tags Hospitals
// @Accept json
// @Produce json
// @Param request body nearbyHospitalReq true "Latitude, Longitude and Radius (km)"
// @Success 200 {array} hospital.Hospital
// @Failure 400 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /hospitals/nearby [post]
func (h *HospitalHandler) FindNearbyHospitals(c *gin.Context) {
	var req nearbyHospitalReq
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	results, err := h.service.FindNearbyHospitals(req.Latitude, req.Longitude, req.RadiusKm)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Nearby hospitals retrieved successfully", results))
}

// ---------------- Get Hospital By Slug ----------------
// @Summary Get hospital by slug
// @Description Retrieve hospital record using hospital slug
// @Tags Hospitals
// @Produce json
// @Param slug path string true "Hospital Slug"
// @Success 200 {object} utils.APIResponse
// @Failure 404 {object} utils.APIResponse
// @Router /hospitals/slug/{slug} [get]
func (h *HospitalHandler) GetHospitalBySlug(c *gin.Context) {
	slug := c.Param("slug")
	hospitalData, err := h.service.GetHospitalBySlug(slug)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Hospital not found"))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Hospital retrieved successfully", hospitalData))
}
