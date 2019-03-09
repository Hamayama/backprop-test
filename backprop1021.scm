;; -*- coding: utf-8 -*-
;;
;; backprop1021.scm
;; 2019-3-9 v1.12
;;
;; ＜内容＞
;;   Gauche を使って、バックプロパゲーションによる学習を行うプログラムです。
;;   出典は以下になります。
;;   「はじめてのディープラーニング」 我妻幸長 SB Creative 2018
;;     (「5.9 バックプロパゲーションの実装 -回帰-」)
;;
;;   詳細については、以下のページを参照ください。
;;   https://github.com/Hamayama/backprop-test
;;
;; ＜流用ベース＞
;;   backprop1001.scm
;;
;; ＜変更点＞
;;   活性化関数を ReLU 関数にした。
;;
(add-load-path "." :relative)
(use gauche.sequence)  ; shuffle
(use gauche.generator) ; generator->list
(use gauche.uvector)
(use gauche.array)
(use math.const)
(use data.random)      ; random-data-seed,reals-normal$
(use srfi-27)          ; random-source-randomize!,default-random-source
(use f64arraysub)      ; 補助モジュール

;; 乱数の初期化
(set! (random-data-seed) (sys-time)) ; data.random (for reals-normal$)
(random-source-randomize! default-random-source) ; srfi-27 (for shuffle)
;; 正規分布の乱数ジェネレータ
(define gen-normal (reals-normal$))


(define outfile        "backprop_result1021.txt") ; 出力ファイル名

(define input-data-0   (lrange 0 2pi 0.1))      ; 入力(リスト)
(define correct-data-0 (map %sin input-data-0)) ; 正解(リスト)
(define n-data         (length input-data-0))   ; データ数

(define input-data     (apply f64array-simple   ; 入力(行列(1 x n-data))
                              0 1 0 n-data
                              ;; (入力の範囲を -1.0～1.0 に変換)
                              (map (lambda (x1) (/ (- x1 pi) pi)) input-data-0)))
(define correct-data   (apply f64array-simple   ; 正解(行列(1 x n-data))
                              0 1 0 n-data
                              correct-data-0))

(define n-in           1)    ; 入力層のニューロン数
(define n-mid          3)    ; 中間層のニューロン数
(define n-out          1)    ; 出力層のニューロン数

(define wb-width       0.01) ; 重みとバイアスの幅
(define eta            0.1)  ; 学習係数
(define epoch          2001) ; エポック数
(define interval       200)  ; 経過の表示間隔


;; 中間層クラス
(define-class <middle-layer> ()
  ((w      :init-value #f) ; 重み          (行列(n-upper x n))
   (b      :init-value #f) ; バイアス      (行列(1 x n))
   (x      :init-value #f) ; 入力          (行列(1 x n-upper))
   (y      :init-value #f) ; 出力          (行列(1 x n))
   (grad-w :init-value #f) ; 重みの勾配    (行列(サイズはwと同じ))
   (grad-b :init-value #f) ; バイアスの勾配(行列(サイズはbと同じ))
   (grad-x :init-value #f) ; 入力の勾配    (行列(サイズはxと同じ))
   ))
(define (middle-layer-init ml n-upper n)
  (slot-set! ml 'w (apply f64array-simple
                          0 n-upper 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal (* n-upper n)))))
  (slot-set! ml 'b (apply f64array-simple
                          0 1 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal n)))))
(define (middle-layer-forward ml x)
  (slot-set! ml 'x x)
  (let1 u (f64array-add-elements (f64array-mul x (slot-ref ml 'w)) (slot-ref ml 'b))
    (slot-set! ml 'y (f64array-relu u)))) ; ReLU関数 ( (max 0 u) )
(define (middle-layer-backward ml grad-y)
  (let1 delta (f64array-mul-elements      ; ReLU関数の微分 ( ステップ関数 )
               grad-y
               (f64array-step (slot-ref ml 'y)))
    (slot-set! ml 'grad-w (f64array-mul (f64array-transpose (slot-ref ml 'x)) delta))
    (slot-set! ml 'grad-b delta)
    (slot-set! ml 'grad-x (f64array-mul delta (f64array-transpose (slot-ref ml 'w))))))
(define (middle-layer-update ml eta)
  (slot-set! ml 'w (f64array-sub-elements
                    (slot-ref ml 'w)
                    (f64array-mul-elements (slot-ref ml 'grad-w) eta)))
  (slot-set! ml 'b (f64array-sub-elements
                    (slot-ref ml 'b)
                    (f64array-mul-elements (slot-ref ml 'grad-b) eta))))


;; 出力層クラス
(define-class <output-layer> ()
  ((w      :init-value #f) ; 重み          (行列(n-upper x n))
   (b      :init-value #f) ; バイアス      (行列(1 x n))
   (x      :init-value #f) ; 入力          (行列(1 x n-upper))
   (y      :init-value #f) ; 出力          (行列(1 x n))
   (grad-w :init-value #f) ; 重みの勾配    (行列(サイズはwと同じ))
   (grad-b :init-value #f) ; バイアスの勾配(行列(サイズはbと同じ))
   (grad-x :init-value #f) ; 入力の勾配    (行列(サイズはxと同じ))
   ))
(define (output-layer-init ol n-upper n)
  (slot-set! ol 'w (apply f64array-simple
                          0 n-upper 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal (* n-upper n)))))
  (slot-set! ol 'b (apply f64array-simple
                          0 1 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal n)))))
(define (output-layer-forward ol x)
  (slot-set! ol 'x x)
  (let1 u (f64array-add-elements (f64array-mul x (slot-ref ol 'w)) (slot-ref ol 'b))
    (slot-set! ol 'y u))) ; 恒等関数
(define (output-layer-backward ol t)
  (let1 delta (f64array-sub-elements (slot-ref ol 'y) t)
    (slot-set! ol 'grad-w (f64array-mul (f64array-transpose (slot-ref ol 'x)) delta))
    (slot-set! ol 'grad-b delta)
    (slot-set! ol 'grad-x (f64array-mul delta (f64array-transpose (slot-ref ol 'w))))))
(define (output-layer-update ol eta)
  (slot-set! ol 'w (f64array-sub-elements
                    (slot-ref ol 'w)
                    (f64array-mul-elements (slot-ref ol 'grad-w) eta)))
  (slot-set! ol 'b (f64array-sub-elements
                    (slot-ref ol 'b)
                    (f64array-mul-elements (slot-ref ol 'grad-b) eta))))


;; メイン処理
(define (main args)
  (define ml (make <middle-layer>))
  (define ol (make <output-layer>))

  ;; 各層の初期化
  (middle-layer-init ml n-in  n-mid)
  (output-layer-init ol n-mid n-out)

  ;; 学習
  (dotimes (i epoch)
    (let ((index-random (shuffle (iota n-data)))
          (total-error  0)
          (result       '()))
      (dolist (idx index-random)
        (let ((x (f64array-ref input-data   0 idx))
              (t (f64array-ref correct-data 0 idx))
              (y #f))
          ;; 順伝播
          (middle-layer-forward  ml (f64array-simple 0 1 0 1 x))
          (output-layer-forward  ol (slot-ref ml 'y))
          ;; 逆伝播 (ouput -> middle の順なので注意)
          (output-layer-backward ol (f64array-simple 0 1 0 1 t))
          (middle-layer-backward ml (slot-ref ol 'grad-x))
          ;; 重みとバイアスの更新
          (middle-layer-update   ml eta)
          (output-layer-update   ol eta)
          ;; 結果の収集
          (when (= (modulo i interval) 0)
            (set! y (f64array-ref (slot-ref ol 'y) 0 0))
            (inc! total-error (* 0.5 (- y t) (- y t)))
            (push! result (cons x y)))
          ))
      ;; 結果の表示
      (when (= (modulo i interval) 0)
        ;(print (sort result < car))
        (print "Epoch: " i " / " epoch)
        (print "Error: " (/. total-error n-data)))
      ;; 最終結果((x,y)の組のデータ)の出力
      (when (= i (- epoch 1))
        (with-output-to-file outfile
          (lambda ()
            (for-each
             (lambda (r) (print (car r) "\t" (cdr r)))
             (sort result < car)))))
      ))

  ;; 最終パラメータの表示
  ;(print "ml.w = "(slot-ref ml 'w))
  ;(print "ml.b = "(slot-ref ml 'b))
  ;(print "ol.w = "(slot-ref ol 'w))
  ;(print "ol.b = "(slot-ref ol 'b))

  0)

