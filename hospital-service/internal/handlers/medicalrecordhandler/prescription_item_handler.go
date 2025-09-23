package medicalrecordhandler

import (
	"net/http"

	"hospital-service/internal/config"
	"hospital-service/internal/models/medicalrecord"
	"hospital-service/internal/services/medicalrecordservice"
	"hospital-service/internal/utils"

	"github.com/gin-gonic/gin"
)

type PrescriptionItemHandler struct {
	service *medicalrecordservice.PrescriptionItemService
	cfg     config.Config

}

// NewPrescriptionItemHandler khởi tạo handler
func NewPrescriptionItemHandler(cfg config.Config, service *medicalrecordservice.PrescriptionItemService) *PrescriptionItemHandler {
	return &PrescriptionItemHandler{service: service, cfg: cfg}
}

//
// -------------------- Create Prescription Item --------------------
// @Summary Create a prescription item
// @Description Add a new prescription item to an existing prescription
// @Tags PrescriptionItems
// @Accept json
// @Produce json
// @Param prescription_item body medicalrecord.PrescriptionItem true "Prescription Item payload"
// @Success 201 {object} medicalrecord.PrescriptionItem
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /prescription_items [post]
//
func (h *PrescriptionItemHandler) CreatePrescriptionItem(c *gin.Context) {
	var req struct {
		PrescriptionID string `json:"prescription_id" binding:"required"`
		DrugName       string `json:"drug_name" binding:"required"`
		Dosage         string `json:"dosage" binding:"required"`
		Frequency      string `json:"frequency" binding:"required"`
		DurationDays   int    `json:"duration_days"`
	}

	// Validate JSON payload
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	// Gọi service
	item, err := h.service.CreatePrescriptionItem(req.PrescriptionID, req.DrugName, req.Dosage, req.Frequency, req.DurationDays)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Prescription item created successfully", item))
}

//
// -------------------- Update Prescription Item --------------------
// @Summary Update a prescription item
// @Description Update details of a specific prescription item
// @Tags PrescriptionItems
// @Accept json
// @Produce json
// @Param item_id path string true "Prescription Item ID"
// @Param prescription_item body medicalrecord.PrescriptionItem true "Prescription Item payload"
// @Success 200 {object} medicalrecord.PrescriptionItem
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /prescription_items/{item_id} [put]
//
func (h *PrescriptionItemHandler) UpdatePrescriptionItem(c *gin.Context) {
	itemID := c.Param("item_id")
	if itemID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "item_id is required"))
		return
	}

	var req medicalrecord.PrescriptionItem
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	req.ItemID = itemID

	if err := h.service.UpdatePrescriptionItem(&req); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Prescription item updated successfully", req))
}

//
// -------------------- Delete Prescription Item --------------------
// @Summary Delete a prescription item
// @Description Remove a specific prescription item by ID
// @Tags PrescriptionItems
// @Produce json
// @Param item_id path string true "Prescription Item ID"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /prescription_items/{item_id} [delete]
//
func (h *PrescriptionItemHandler) DeletePrescriptionItem(c *gin.Context) {
	itemID := c.Param("item_id")
	if itemID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "item_id is required"))
		return
	}

	if err := h.service.DeletePrescriptionItem(itemID); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Prescription item deleted successfully", nil))
}
