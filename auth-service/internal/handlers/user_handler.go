package handlers

import (
	"auth-service/internal/services"
	"auth-service/internal/utils"
	"net/http"

	"github.com/gin-gonic/gin"
)

type UserHandler struct {
	service *services.UserService
}

func NewUserHandler(service *services.UserService) *UserHandler {
	return &UserHandler{service: service}
}

type createUserReq struct {
	Username    string `json:"username" example:"alice" binding:"required"`
	Email       string `json:"email" example:"alice@example.com" binding:"required,email"`
	Password    string `json:"password" example:"secret" binding:"required"`
	FirebaseUID string `json:"firebase_uid,omitempty" example:""` // optional
	Role        string `json:"role" example:"patient" binding:"required"` // patient/doctor/admin
}

type updateUserReq struct {
	Username    *string `json:"username,omitempty" example:"alice"`
	Email       *string `json:"email,omitempty" example:"alice@example.com"`
	Password    *string `json:"password,omitempty" example:"newsecret"`
	FirebaseUID *string `json:"firebase_uid,omitempty" example:""` // optional
	Role        *string `json:"role,omitempty" example:"patient"` // patient/doctor/admin
}

// @Summary Create user
// @Tags Users
// @Accept json
// @Produce json
// @Param user body createUserReq true "User data"
// @Success 201 {object} utils.APIResponse
// @Router /private/users [post]
func (h *UserHandler) Create(c *gin.Context) {
	var req createUserReq
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	// firebase_uid optional
	firebaseUID := req.FirebaseUID
	if firebaseUID == "" {
		firebaseUID = ""
	}

	u, err := h.service.CreateUser(req.Username, req.Email, req.Password, firebaseUID, req.Role)
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}
	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "User created", u))
}

// @Summary Get user
// @Tags Users
// @Produce json
// @Param id path string true "User ID"
// @Success 200 {object} utils.APIResponse
// @Router /private/users/{id} [get]
func (h *UserHandler) Get(c *gin.Context) {
	id := c.Param("id")
	u, err := h.service.GetUser(id)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "User found", u))
}

// @Summary Update user
// @Tags Users
// @Accept json
// @Produce json
// @Param id path string true "User ID"
// @Param user body updateUserReq true "User updates"
// @Success 200 {object} utils.APIResponse
// @Router /private/users/{id} [put]
func (h *UserHandler) Update(c *gin.Context) {
	id := c.Param("id")
	var req updateUserReq
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	// Chuyển struct thành map[string]interface{} cho service
	updates := make(map[string]interface{})
	if req.Username != nil {
		updates["username"] = *req.Username
	}
	if req.Email != nil {
		updates["email"] = *req.Email
	}
	if req.Password != nil {
		updates["password"] = *req.Password
	}
	if req.FirebaseUID != nil {
		updates["firebase_uid"] = *req.FirebaseUID
	}
	if req.Role != nil {
		updates["role"] = *req.Role
	}

	u, err := h.service.UpdateUser(id, updates)
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "User updated", u))
}

// @Summary Delete user
// @Tags Users
// @Produce json
// @Param id path string true "User ID"
// @Success 200 {object} utils.APIResponse
// @Router /private/users/{id} [delete]
func (h *UserHandler) Delete(c *gin.Context) {
	id := c.Param("id")
	if err := h.service.DeleteUser(id); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "User deleted", nil))
}

// @Summary List users
// @Tags Users
// @Produce json
// @Success 200 {object} utils.APIResponse
// @Router /private/users [get]
func (h *UserHandler) List(c *gin.Context) {
	users, err := h.service.ListUsers()
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Users retrieved", users))
}
