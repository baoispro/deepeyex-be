package handlers

import (
	"auth-service/internal/config"
	"auth-service/internal/services"
	"auth-service/internal/utils"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

type AuthHandler struct {
	service *services.AuthService
	cfg     config.Config
}

func NewAuthHandler(cfg config.Config, service *services.AuthService) *AuthHandler {
	return &AuthHandler{service: service, cfg: cfg}
}

type registerReq struct {
	Username string `json:"username" example:"alice" binding:"required"`
	Password string `json:"password" example:"secret" binding:"required"`
	Email    string `json:"email" example:"nguyena@gmail.com" binding:"required"`
}
type loginReq struct {
	Username string `json:"username" example:"alice" binding:"required"`
	Password string `json:"password" example:"secret" binding:"required"`
}
type loginFirebaseReq struct {
	FirebaseUID string `json:"firebase_uid" binding:"required"`
	Email       string `json:"email" binding:"required,email"`
}
type tokenRes struct {
	AccessToken  string    `json:"access_token"`
	AccessExpire time.Time `json:"access_expire"`
	UserID       string    `json:"user_id,omitempty"`
	Role         string    `json:"role,omitempty"`
}

// ----- helpers cookie -----
func (h *AuthHandler) setRefreshCookie(c *gin.Context, token string, expires time.Time) {
	c.SetCookie(
		h.cfg.RefreshCookieName,
		token,
		int(time.Until(expires).Seconds()),
		"/",
		h.cfg.CookieDomain,
		h.cfg.CookieSecure,
		true,
	)
}

func (h *AuthHandler) clearRefreshCookie(c *gin.Context) {
	c.SetCookie(h.cfg.RefreshCookieName, "", -1, "/", h.cfg.CookieDomain, h.cfg.CookieSecure, true)
}

// Register
// Register godoc
// @Summary Register a new user
// @Description Create a new account
// @Tags Public
// @Accept json
// @Produce json
// @Param user body registerReq true "Register request"
// @Success 201 {object} map[string]string "User registered"
// @Failure 400 {object} map[string]string "Invalid payload or registration failed"
// @Router /public/register [post]
func (h *AuthHandler) Register(c *gin.Context) {
	var req registerReq
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}
	if err := h.service.Register(req.Username, req.Email, req.Password); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}
	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "User registered successfully", nil))
}

// Login
// Login godoc
// @Summary Login user
// @Description Authenticate user and return access/refresh tokens
// @Tags Public
// @Accept json
// @Produce json
// @Param login body loginReq true "Login request"
// @Success 200 {object} tokenRes
// @Failure 401 {object} map[string]string "Invalid credentials"
// @Router /public/login [post]
func (h *AuthHandler) Login(c *gin.Context) {
	var req loginReq
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusUnauthorized, utils.ErrorResponse(http.StatusUnauthorized, "invalid login payload"))
		return
	}
	access, aExp, refresh, rExp, u, err := h.service.Login(req.Username, req.Password)
	if err != nil {
		c.JSON(http.StatusUnauthorized, utils.ErrorResponse(http.StatusUnauthorized, err.Error()))
		return
	}
	h.setRefreshCookie(c, refresh, rExp)
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Login successful", tokenRes{
		AccessToken:  access,
		AccessExpire: aExp,
		UserID:       u.ID,
		Role:         string(u.Role),
	}))
}

// Login firebase
// LoginFirebase godoc
// @Summary Login with Firebase
// @Description Authenticate user using Firebase UID & email, then return access/refresh tokens
// @Tags Public
// @Accept json
// @Produce json
// @Param loginFirebase body loginFirebaseReq true "Firebase login request"
// @Success 200 {object} tokenRes
// @Failure 401 {object} map[string]string "Invalid firebase payload or login failed"
// @Router /public/login/firebase [post]
func (h *AuthHandler) LoginFirebase(c *gin.Context) {
	var req loginFirebaseReq
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusUnauthorized, utils.ErrorResponse(http.StatusUnauthorized, "invalid firebase login payload"))
		return
	}

	access, aExp, refresh, rExp, u, err := h.service.LoginFirebase(req.FirebaseUID, req.Email)
	if err != nil {
		c.JSON(http.StatusUnauthorized, utils.ErrorResponse(http.StatusUnauthorized, err.Error()))
		return
	}
	h.setRefreshCookie(c, refresh, rExp)
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Firebase login successful", tokenRes{
		AccessToken:  access,
		AccessExpire: aExp,
		UserID:       u.ID,
		Role:         string(u.Role),
	}))
}

// Refresh
// @Summary Refresh access token
// @Description Generate new access token using refresh cookie
// @Tags Public
// @Produce json
// @Success 200 {object} tokenRes
// @Failure 401 {object} map[string]string "Missing or invalid refresh token"
// @Router /public/refresh [post]
func (h *AuthHandler) Refresh(c *gin.Context) {
	refreshCookie, err := c.Cookie(h.cfg.RefreshCookieName)
	if err != nil || refreshCookie == "" {
		c.JSON(http.StatusUnauthorized, utils.ErrorResponse(http.StatusUnauthorized, "missing refresh token"))
		return
	}

	access, aExp, newRefresh, newExp, err := h.service.Refresh(refreshCookie)
	if err != nil {
		c.JSON(http.StatusUnauthorized, utils.ErrorResponse(http.StatusUnauthorized, err.Error()))
		return
	}
	h.setRefreshCookie(c, newRefresh, newExp)
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Token refreshed successfully", tokenRes{AccessToken: access, AccessExpire: aExp}))
}

// Logout
// Logout godoc
// @Summary Logout user
// @Description Clear refresh cookie and invalidate refresh token
// @Tags Public
// @Produce json
// @Success 200 {object} map[string]string "Logged out"
// @Router /public/logout [post]
func (h *AuthHandler) Logout(c *gin.Context) {
	if refresh, err := c.Cookie(h.cfg.RefreshCookieName); err == nil && refresh != "" {
		h.service.Logout(refresh)
	}
	h.clearRefreshCookie(c)
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Logged out successfully", nil))
}

// Me
// Me godoc
// @Summary Get current user info
// @Description Return user ID and role from JWT claims
// @Tags Private
// @Security BearerAuth
// @Produce json
// @Success 200 {object} map[string]string "User info"
// @Router /private/me [get]
func (h *AuthHandler) Me(c *gin.Context) {
	uid := c.GetString("uid")
	role := c.GetString("role")
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "User info retrieved successfully", map[string]string{
		"user_id": uid,
		"role":    role,
	}))
}
