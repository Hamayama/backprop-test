;; -*- coding: utf-8 -*-
;;
;; backprop_main.scm
;; 2019-6-17 v2.52
;;
;; ＜内容＞
;;   Gauche を使って、バックプロパゲーションによる学習を行うプログラムです。
;;   出典は以下になります。
;;   「はじめてのディープラーニング」 我妻幸長 SB Creative 2018
;;     (「5.9 バックプロパゲーションの実装 -回帰-」)
;;
;;   ただし、高速化のために、計算用の行列の追加や、複合演算命令への変更を行っています。
;;   また、中間層の数の設定と、活性化関数 (シグモイド / ReLU / tanh) の選択を可能にしています。
;;
;;   詳細については、以下のページを参照ください。
;;   https://github.com/Hamayama/backprop-test
;;
;; ＜使い方＞
;;   gosh backprop_main.scm paramfile
;;     paramfile : パラメータ設定ファイル
;;
(add-load-path "." :relative)
(use gauche.sequence)  ; shuffle
(use gauche.generator) ; generator->list
(use gauche.uvector)
(use gauche.array)
(use math.const)
(use data.random)      ; random-data-seed,reals-normal$
(use srfi-27)          ; random-source-randomize!,default-random-source
(use f2arrmat)         ; 2次元の f64array を扱う

;; 乱数の初期化
(set! (random-data-seed) (sys-time)) ; data.random (for reals-normal$)
(random-source-randomize! default-random-source) ; srfi-27 (for shuffle)
;; 正規分布の乱数ジェネレータ
(define gen-normal (reals-normal$))
;; データ(リスト)の範囲変換 (minx1～maxx1 を minx2～maxx2 に変換する)
(define (range-conv data minx1 maxx1 minx2 maxx2)
  (map (lambda (x1) (+ minx2 (/. (* (- x1 minx1) (- maxx2 minx2)) (- maxx1 minx1))))
       data))


(define outfile        "backprop_result.txt")   ; 出力ファイル名

(define input-data-0   (lrange 0 2pi 0.1))      ; 入力(リスト)
(define correct-data-0 (map %sin input-data-0)) ; 正解(リスト)
(define n-data         (length input-data-0))   ; データ数

(define input-data     (apply f2-array          ; 入力(行列(1 x n-data))
                              0 1 0 n-data
                              ;; (入力の範囲を -1.0～1.0 に変換)
                              (range-conv input-data-0 0 2pi -1.0 1.0)))
(define correct-data   (apply f2-array          ; 正解(行列(1 x n-data))
                              0 1 0 n-data
                              correct-data-0))

(define n-in           1)     ; 入力層のニューロン数
(define n-mid          3)     ; 中間層のニューロン数
(define n-out          1)     ; 出力層のニューロン数

(define ml-num         1)     ; 中間層の数
(define ml-func        'tanh) ; 中間層の活性化関数の選択(sigmoid / relu / tanh)

(define wb-width       0.01)  ; 重みとバイアスの幅
(define eta            0.1)   ; 学習係数
(define epoch          2001)  ; エポック数
(define interval       200)   ; 経過の表示間隔


;; 中間層の活性化関数の生成
(define ml-act-func  #f) ; 中間層の活性化関数
(define ml-diff-func #f) ; 中間層の活性化関数の微分
(define (ml-func-init)
  (set! ml-act-func
        (ecase ml-func
          ((sigmoid)
           ;; シグモイド関数 : y = 1 / (1 + exp(-u))
           ;(lambda (y u) (f2-array-sigmoid! y u)))
           f2-array-sigmoid!)
          ((relu)
           ;; ReLU関数 : y = max(0, u)
           ;(lambda (y u) (f2-array-relu! y u)))
           f2-array-relu!)
          ((tanh)
           ;; tanh関数 : y = tanh(u)
           ;(lambda (y u) (f2-array-tanh! y u)))
           f2-array-tanh!)
          ))
  (set! ml-diff-func
        (ecase ml-func
          ((sigmoid)
           ;; シグモイド関数の微分 : delta = (1 - y) * y
           (lambda (delta y)
             (f2-array-mul-elements! delta (f2-array-sub-elements! delta y 1) -1 y)))
          ((relu)
           ;; ReLU関数の微分 (ステップ関数) : delta = (y > 0 ? 1 : 0)
           ;(lambda (delta y) (f2-array-step! delta y)))
           f2-array-step!)
          ((tanh)
           ;; tanh関数の微分 : delta = 1 - y * y
           (lambda (delta y)
             (f2-array-mul-elements!
              delta (f2-array-sub-elements! delta (f2-array-pow! delta y 2) 1) -1)))
          ))
  )


;; 中間層クラス
(define-class <middle-layer> ()
  ((w      :init-value #f) ; 重み          (行列(n-upper x n))
   (b      :init-value #f) ; バイアス      (行列(1 x n))
   (x      :init-value #f) ; 入力          (行列(1 x n-upper))
   (y      :init-value #f) ; 出力          (行列(1 x n))
   (grad-w :init-value #f) ; 重みの勾配    (行列(サイズはwと同じ))
   (grad-b :init-value #f) ; バイアスの勾配(行列(サイズはbと同じ))
   (grad-x :init-value #f) ; 入力の勾配    (行列(サイズはxと同じ))
   (u      :init-value #f) ; 計算用        (行列(サイズはyと同じ))
   (delta  :init-value #f) ; 計算用        (行列(サイズはyと同じ))
   ))
(define (middle-layer-init ml n-upper n)
  (slot-set! ml 'w (apply f2-array
                          0 n-upper 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal (* n-upper n)))))
  (slot-set! ml 'b (apply f2-array
                          0 1 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal n))))
  (slot-set! ml 'x      (make-f2-array 0 1 0 n-upper))
  (slot-set! ml 'y      (make-f2-array 0 1 0 n))
  (slot-set! ml 'grad-w (make-f2-array-same-shape (slot-ref ml 'w)))
  (slot-set! ml 'grad-b (make-f2-array-same-shape (slot-ref ml 'b)))
  (slot-set! ml 'grad-x (make-f2-array-same-shape (slot-ref ml 'x)))
  (slot-set! ml 'u      (make-f2-array-same-shape (slot-ref ml 'y)))
  (slot-set! ml 'delta  (make-f2-array-same-shape (slot-ref ml 'y)))
  )
(define (middle-layer-forward ml x)
  (slot-set! ml 'x x)
  ;; u = x * w + b
  (f2-array-copy!   (slot-ref ml 'u) (slot-ref ml 'b))
  (f2-array-ab+c! x (slot-ref ml 'w) (slot-ref ml 'u) 1.0 1.0 #f #f)
  ;; y = ml-act-func(u)
  (ml-act-func      (slot-ref ml 'y) (slot-ref ml 'u))
  )
(define (middle-layer-backward ml grad-y)
  ;; delta = grad-y * ml-diff-func(y)
  (f2-array-mul-elements!
   (slot-ref ml 'delta) grad-y (ml-diff-func (slot-ref ml 'delta) (slot-ref ml 'y)))
  ;; grad-w = tx * delta : (ただし tx は x の転置行列)
  (f2-array-ab+c! (slot-ref ml 'x) (slot-ref ml 'delta) (slot-ref ml 'grad-w) 1.0 0.0 #t #f)
  ;; grad-b = delta
  (slot-set! ml 'grad-b (slot-ref ml 'delta))
  ;; grad-x = delta * tw : (ただし tw は w の転置行列)
  (f2-array-ab+c! (slot-ref ml 'delta) (slot-ref ml 'w) (slot-ref ml 'grad-x) 1.0 0.0 #f #t)
  )
(define (middle-layer-update ml eta)
  ;; w -= eta * grad-w
  (f2-array-ra+b! (- eta) (slot-ref ml 'grad-w) (slot-ref ml 'w))
  ;; b -= eta * grad-b
  (f2-array-ra+b! (- eta) (slot-ref ml 'grad-b) (slot-ref ml 'b))
  )


;; 出力層クラス
(define-class <output-layer> ()
  ((w      :init-value #f) ; 重み          (行列(n-upper x n))
   (b      :init-value #f) ; バイアス      (行列(1 x n))
   (x      :init-value #f) ; 入力          (行列(1 x n-upper))
   (y      :init-value #f) ; 出力          (行列(1 x n))
   (grad-w :init-value #f) ; 重みの勾配    (行列(サイズはwと同じ))
   (grad-b :init-value #f) ; バイアスの勾配(行列(サイズはbと同じ))
   (grad-x :init-value #f) ; 入力の勾配    (行列(サイズはxと同じ))
   (u      :init-value #f) ; 計算用        (行列(サイズはyと同じ))
   (delta  :init-value #f) ; 計算用        (行列(サイズはyと同じ))
   (t      :init-value #f) ; 計算用        (行列(サイズはyと同じ))
   ))
(define (output-layer-init ol n-upper n)
  (slot-set! ol 'w (apply f2-array
                          0 n-upper 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal (* n-upper n)))))
  (slot-set! ol 'b (apply f2-array
                          0 1 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal n))))
  (slot-set! ol 'x      (make-f2-array 0 1 0 n-upper))
  (slot-set! ol 'y      (make-f2-array 0 1 0 n))
  (slot-set! ol 'grad-w (make-f2-array-same-shape (slot-ref ol 'w)))
  (slot-set! ol 'grad-b (make-f2-array-same-shape (slot-ref ol 'b)))
  (slot-set! ol 'grad-x (make-f2-array-same-shape (slot-ref ol 'x)))
  (slot-set! ol 'u      (make-f2-array-same-shape (slot-ref ol 'y)))
  (slot-set! ol 'delta  (make-f2-array-same-shape (slot-ref ol 'y)))
  (slot-set! ol 't      (make-f2-array-same-shape (slot-ref ol 'y)))
  )
(define (output-layer-forward ol x)
  (slot-set! ol 'x x)
  ;; u = x * w + b
  (f2-array-copy!   (slot-ref ol 'u) (slot-ref ol 'b))
  (f2-array-ab+c! x (slot-ref ol 'w) (slot-ref ol 'u) 1.0 1.0 #f #f)
  ;; 恒等関数 : y = u
  (slot-set! ol 'y (slot-ref ol 'u))
  )
(define (output-layer-backward ol t)
  (slot-set! ol 't t)
  ;; delta = y - t
  (f2-array-sub-elements! (slot-ref ol 'delta) (slot-ref ol 'y) t)
  ;; grad-w = tx * delta : (ただし tx は x の転置行列)
  (f2-array-ab+c! (slot-ref ol 'x) (slot-ref ol 'delta) (slot-ref ol 'grad-w) 1.0 0.0 #t #f)
  ;; grad-b = delta
  (slot-set! ol 'grad-b (slot-ref ol 'delta))
  ;; grad-x = delta * tw : (ただし tw は w の転置行列)
  (f2-array-ab+c! (slot-ref ol 'delta) (slot-ref ol 'w) (slot-ref ol 'grad-x) 1.0 0.0 #f #t)
  )
(define (output-layer-update ol eta)
  ;; w -= eta * grad-w
  (f2-array-ra+b! (- eta) (slot-ref ol 'grad-w) (slot-ref ol 'w))
  ;; b -= eta * grad-b
  (f2-array-ra+b! (- eta) (slot-ref ol 'grad-b) (slot-ref ol 'b))
  )


;; メイン処理
(define (main args)
  (define paramfile (list-ref args 1 #f))
  (define mls       #f) ; 中間層の配列(ベクタ)
  (define mls-rev   #f) ; 中間層の逆順の配列(ベクタ)
  (define ol        #f) ; 出力層

  ;; パラメータ設定ファイルのロード
  (if paramfile (load paramfile))

  ;; 各層の生成
  (set! mls     (vector-tabulate ml-num (lambda (ml) (make <middle-layer>))))
  (set! mls-rev (list->vector (reverse (vector->list mls))))
  (set! ol      (make <output-layer>))

  ;; 各層の初期化
  (for-each-with-index
   (lambda (i ml)
     (if (= i 0)
       (middle-layer-init ml n-in  n-mid)
       (middle-layer-init ml n-mid n-mid)))
   mls)
  (output-layer-init ol n-mid n-out)
  (ml-func-init)

  ;; 学習
  (dotimes (i epoch)
    (let ((index-random (shuffle (iota n-data)))
          (total-error  0)
          (result       '()))
      (dolist (idx index-random)
        (let ((x (f2-array-ref input-data   0 idx))
              (t (f2-array-ref correct-data 0 idx))
              (y #f))
          ;; 順伝播
          (f2-array-set! (slot-ref (vector-ref mls 0) 'x) 0 0 x)
          (for-each-with-index
           (lambda (i ml)
             (if (= i 0)
               (middle-layer-forward ml (slot-ref (vector-ref mls 0) 'x))
               (middle-layer-forward ml (slot-ref (vector-ref mls (- i 1)) 'y))))
           mls)
          (output-layer-forward  ol (slot-ref (vector-ref mls (- ml-num 1)) 'y))
          ;; 逆伝播 (ouput -> middle の順なので注意)
          (f2-array-set! (slot-ref ol 't) 0 0 t)
          (output-layer-backward ol (slot-ref ol 't))
          (for-each-with-index
           (lambda (i ml)
             (if (= i 0)
               (middle-layer-backward ml (slot-ref ol 'grad-x))
               (middle-layer-backward ml (slot-ref (vector-ref mls-rev (- i 1)) 'grad-x))))
           mls-rev)
          ;; 重みとバイアスの更新
          (for-each (lambda (ml) (middle-layer-update ml eta)) mls)
          (output-layer-update ol eta)
          ;; 結果の収集
          (when (= (modulo i interval) 0)
            (set! y (f2-array-ref (slot-ref ol 'y) 0 0))
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
  ;(for-each-with-index
  ; (lambda (i ml)
  ;   (format #t "ml~D.w = ~A~%" i (slot-ref (vector-ref mls i) 'w))
  ;   (format #t "ml~D.b = ~A~%" i (slot-ref (vector-ref mls i) 'b)))
  ; mls)
  ;(print "ol.w = "(slot-ref ol 'w))
  ;(print "ol.b = "(slot-ref ol 'b))

  0)

