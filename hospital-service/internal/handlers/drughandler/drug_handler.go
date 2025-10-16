package drughandler

import (
	"hospital-service/internal/config"
	"hospital-service/internal/services/drugservice"
	"hospital-service/internal/utils"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
)

type DrugHandler struct {
	service *drugservice.DrugService
	cfg     config.Config
}

func NewDrugHandler(cfg config.Config, service *drugservice.DrugService) *DrugHandler {
	return &DrugHandler{service: service, cfg: cfg}
}

// ---------------- Create Drug ----------------
// @Summary Create a new drug
// @Description Add a new drug record with optional image upload
// @Tags Drugs
// @Accept multipart/form-data
// @Produce json
// @Param name formData string true "Drug name"
// @Param description formData string false "Description"
// @Param price formData number true "Price"
// @Param stock formData int true "Stock quantity"
// @Param discount formData number false "Discount percent"
// @Param image formData file false "Drug image"
// @Success 201 {object} drug.Drug
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /drugs [post]
func (h *DrugHandler) CreateDrug(c *gin.Context) {
	name := c.PostForm("name")
	description := c.PostForm("description")
	priceStr := c.PostForm("price")
	stockStr := c.PostForm("stock_quantity")
	discountStr := c.PostForm("discount_percent")

	// Parse numeric fields
	price, err := strconv.ParseFloat(priceStr, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid price"))
		return
	}
	stock, err := strconv.Atoi(stockStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid stock quantity"))
		return
	}
	discount := 0.0
	if discountStr != "" {
		discount, err = strconv.ParseFloat(discountStr, 64)
		if err != nil {
			c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid discount"))
			return
		}
	}

	var imageFile interface{}
	fileHeader, err := c.FormFile("image")
	if err == nil {
		imageFile = fileHeader
	}

	d, err := h.service.CreateDrug(name, description, price, stock, discount, imageFile)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Drug created successfully", d))
}

// ---------------- Get Drug By ID ----------------
// @Summary Get drug by ID
// @Description Retrieve drug by drug ID
// @Tags Drugs
// @Produce json
// @Param drug_id path string true "Drug ID"
// @Success 200 {object} drug.Drug
// @Failure 404 {object} map[string]string
// @Router /drugs/{drug_id} [get]
func (h *DrugHandler) GetDrugByID(c *gin.Context) {
	id := c.Param("drug_id")
	d, err := h.service.GetDrug(id)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Drug not found"))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Drug retrieved successfully", d))
}

// ---------------- Update Drug ----------------
// @Summary Update drug info
// @Description Update drug record by ID (with optional image)
// @Tags Drugs
// @Accept multipart/form-data
// @Produce json
// @Param drug_id path string true "Drug ID"
// @Param name formData string false "Drug name"
// @Param description formData string false "Description"
// @Param price formData number false "Price"
// @Param stock formData int false "Stock quantity"
// @Param discount formData number false "Discount percent"
// @Param image formData file false "Drug image"
// @Success 200 {object} drug.Drug
// @Failure 400 {object} map[string]string
// @Failure 404 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /drugs/{drug_id} [put]
func (h *DrugHandler) UpdateDrug(c *gin.Context) {
	id := c.Param("drug_id")
	d, err := h.service.GetDrug(id)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Drug not found"))
		return
	}

	if name := c.PostForm("name"); name != "" {
		d.Name = name
	}
	if description := c.PostForm("description"); description != "" {
		d.Description = description
	}
	if priceStr := c.PostForm("price"); priceStr != "" {
		price, err := strconv.ParseFloat(priceStr, 64)
		if err != nil {
			c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid price"))
			return
		}
		d.Price = price
	}
	if stockStr := c.PostForm("stock"); stockStr != "" {
		stock, err := strconv.Atoi(stockStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid stock quantity"))
			return
		}
		d.StockQuantity = stock
	}
	if discountStr := c.PostForm("discount"); discountStr != "" {
		discount, err := strconv.ParseFloat(discountStr, 64)
		if err != nil {
			c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid discount"))
			return
		}
		d.DiscountPercent = discount
	}

	var imageFile interface{}
	fileHeader, err := c.FormFile("image")
	if err == nil {
		imageFile = fileHeader
	}

	if err := h.service.UpdateDrug(d, imageFile); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Drug updated successfully", d))
}

// ---------------- Delete Drug ----------------
// @Summary Delete drug
// @Description Delete drug by ID
// @Tags Drugs
// @Produce json
// @Param drug_id path string true "Drug ID"
// @Success 200 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /drugs/{drug_id} [delete]
func (h *DrugHandler) DeleteDrug(c *gin.Context) {
	id := c.Param("drug_id")
	if err := h.service.DeleteDrug(id); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Drug deleted successfully", nil))
}

// ---------------- List Drugs ----------------
// @Summary List all drugs
// @Description Retrieve all drugs with optional filters
// @Tags Drugs
// @Produce json
// @Param name query string false "Filter by drug name (partial match)"
// @Param min_price query number false "Filter by minimum price"
// @Param max_price query number false "Filter by maximum price"
// @Param min_stock query int false "Filter by minimum stock quantity"
// @Param max_stock query int false "Filter by maximum stock quantity"
// @Success 200 {array} drug.Drug
// @Failure 500 {object} map[string]string
// @Router /drugs [get]
func (h *DrugHandler) ListDrugs(c *gin.Context) {
	// Lấy query params
	name := c.Query("name")
	minPrice := c.Query("min_price")
	maxPrice := c.Query("max_price")
	minStock := c.Query("min_stock")
	maxStock := c.Query("max_stock")

	list, err := h.service.ListDrugs(name, minPrice, maxPrice, minStock, maxStock)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Drugs retrieved successfully", list))
}
