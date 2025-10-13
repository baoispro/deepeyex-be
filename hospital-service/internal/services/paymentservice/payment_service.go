package paymentservice

import (
	"crypto/hmac"
	"crypto/sha512"
	"encoding/hex"
	"hospital-service/internal/config"
	"net/url"
	"sort"
	"strconv"
	"time"
)

type VnpayService struct {
	tmnCode   string
	secretKey string
	vnpURL    string
	returnURL string
}

func NewVnpayService(cfg config.Config) *VnpayService {
	return &VnpayService{
		tmnCode:   cfg.VnpTmnCode,
		secretKey: cfg.VnpSecretKey,
		vnpURL:    cfg.VnpUrl,
		returnURL: cfg.VnpReturnUrl,
	}
}

func (s *VnpayService) CreatePaymentURL(amount int, orderId string) (string, error) {
	vnpParams := map[string]string{
		"vnp_Version":    "2.1.0",
		"vnp_Command":    "pay",
		"vnp_TmnCode":    s.tmnCode,
		"vnp_Locale":     "vn",
		"vnp_CurrCode":   "VND",
		"vnp_TxnRef":     orderId,
		"vnp_OrderInfo":  "Thanh toan don hang",
		"vnp_OrderType":  "other",
		"vnp_Amount":     strconv.Itoa(amount * 100),
		"vnp_ReturnUrl":  s.returnURL + "?orderId=" + orderId,
		"vnp_IpAddr":     "127.0.0.1",
		"vnp_CreateDate": time.Now().Format("20060102150405"),
	}

	// --- sort keys ---
	keys := make([]string, 0, len(vnpParams))
	for k := range vnpParams {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// --- build query string ENCODED ---
	values := url.Values{}
	for _, k := range keys {
		values.Add(k, vnpParams[k])
	}
	queryString := values.Encode() // đã url-encode giống Node

	// --- signData = queryString ---
	h := hmac.New(sha512.New, []byte(s.secretKey))
	h.Write([]byte(queryString))
	secureHash := hex.EncodeToString(h.Sum(nil))

	// --- return final URL ---
	return s.vnpURL + "?" + queryString + "&vnp_SecureHash=" + secureHash, nil
}

func (s *VnpayService) VerifyReturn(query url.Values) bool {
	vnpSecureHash := query.Get("vnp_SecureHash")
	
	// CHỈ lấy các params bắt đầu với "vnp_" (không lấy orderId hay params khác)
	vnpParams := url.Values{}
	for key, values := range query {
		// Bỏ qua vnp_SecureHash và các params không phải vnp_*
		if key != "vnp_SecureHash" && len(key) > 4 && key[:4] == "vnp_" {
			for _, value := range values {
				vnpParams.Add(key, value)
			}
		}
	}

	// Sort keys
	keys := make([]string, 0, len(vnpParams))
	for k := range vnpParams {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// Build query string theo thứ tự alphabet
	values := url.Values{}
	for _, k := range keys {
		values.Add(k, vnpParams.Get(k))
	}
	queryString := values.Encode()

	// Tính hash
	h := hmac.New(sha512.New, []byte(s.secretKey))
	h.Write([]byte(queryString))
	hash := hex.EncodeToString(h.Sum(nil))



	return hash == vnpSecureHash
}
